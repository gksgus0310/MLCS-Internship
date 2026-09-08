#==========================================#
# Title:  02-004 Linear Model Predictive Control
#         차량 경로추종 MPC. 003의 QP 를 매 스텝 다시 푸는 것이 MPC 다.
#
#         Part 1. 문제 설정 - 어떤 상황인가
#         Part 2. MPC vs LQR (조향 한계가 걸릴 때)
#         Part 3. 예측구간 N 과 종단비용
#         Part 4. 같은 MPC 를 FSAE 차량 파라미터로
# Author: Hyun Han
# Date:   2026-09-08
#==========================================#
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from qpsolvers import solve_qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
summary = {}
SOLVER = "quadprog"


# =========================================================== #
#  차량 모델 (바이시클 모델, 경로 기준 오차 좌표)
#     상태 x = [ey, dey, epsi, depsi]
#       ey    경로 중심선에서 벗어난 횡방향 거리 [m]
#       epsi  경로 접선 대비 차량 방위각 오차 [rad]
#     입력 u = delta (앞바퀴 조향각) [rad],  |delta| <= delta_max
#     외란 d = 곡률항 (psiref = vx / R)
# =========================================================== #
class Vehicle:
    def __init__(self, m, Iz, lf, lr, Cf, Cr, delta_max, name):
        self.m, self.Iz, self.lf, self.lr = m, Iz, lf, lr
        self.Cf, self.Cr, self.delta_max, self.name = Cf, Cr, delta_max, name
        self.nx, self.nu = 4, 1

    def get_lti(self, vx, psiref):
        m, Iz, lf, lr, Cf, Cr = self.m, self.Iz, self.lf, self.lr, self.Cf, self.Cr
        mvx, Izvx, lfCf, lrCr = m*vx, Iz*vx, lf*Cf, lr*Cr
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -2*(Cf+Cr)/mvx,      2*(Cf+Cr)/m,      -2*(lfCf-lrCr)/mvx],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -2*(lfCf-lrCr)/Izvx, 2*(lfCf-lrCr)/Iz, -2*(lf*lfCf-lr*lrCr)/Izvx],
        ])
        B = np.array([[0.0], [2*Cf/m], [0.0], [2*lfCf/Iz]])
        d = np.array([[0.0], [(-2*(lfCf-lrCr)/mvx - vx)*psiref],
                      [0.0], [-2*(lf*lfCf-lr*lrCr)/Izvx*psiref]])
        return A, B, d

    def understeer_gradient(self):
        L = self.lf + self.lr
        return self.m / L * (self.lr/(2*self.Cf) - self.lf/(2*self.Cr))


SEDAN = Vehicle(2979.51, 4549.29, 2.49, 2.75, 33595.26, 57855.95, 0.30, "예제 세단 (3톤)")
# FSAE: 질량·조향각은 MECar 2027 목표치, 관성·코너링강성은 이 급 차량의 통상값 추정
FSAE = Vehicle(300.0, 110.0, 0.72, 0.81, 12000.0, 13000.0, 0.42, "FSAE (300kg)")


class MPC:
    """003 에서 만든 QP 를 매 스텝 다시 푸는 것. 첫 입력만 쓰고 나머지는 버린다."""
    def __init__(self, model, vx, psiref, N, Ts, Q, R, terminal=True):
        Ac, Bc, dc = model.get_lti(vx, psiref)
        nx, nu = model.nx, model.nu
        Ad, Bfull, *_ = cont2discrete(
            (Ac, np.hstack([Bc, dc]), np.eye(nx), np.zeros((nx, nu+1))), Ts)
        Bd, dd = np.hsplit(Bfull, (nu,))
        Pf = solve_discrete_are(Ad, Bd, Q, R) if terminal else Q

        # X = T x0 + S U + t
        S = np.zeros((nx*N, nu*N)); T = np.zeros((nx*N, nx)); t = np.zeros((nx*N, 1))
        for j in range(N):
            T[j*nx:(j+1)*nx] = np.linalg.matrix_power(Ad, j+1)
            for k in range(j+1):
                S[j*nx:(j+1)*nx, k*nu:(k+1)*nu] = np.linalg.matrix_power(Ad, j-k) @ Bd
                t[j*nx:(j+1)*nx] += np.linalg.matrix_power(Ad, k) @ dd
        QS = block_diag(*([Q]*(N-1)), Pf) @ S
        self.P = 2.0*(S.T @ QS + block_diag(*([R]*N)))
        self.P = 0.5*(self.P + self.P.T)
        self.TQS, self.tQS = 2.0*T.T @ QS, 2.0*t.T @ QS
        self.lb = np.full(nu*N, -model.delta_max)
        self.ub = np.full(nu*N,  model.delta_max)
        self.nu = nu
        # 비교용 무한구간 LQR 이득과 곡률 피드포워드
        Pinf = solve_discrete_are(Ad, Bd, Q, R)
        self.Klqr = np.linalg.solve(R + Bd.T @ Pinf @ Bd, Bd.T @ Pinf @ Ad)
        self.u_ff = np.linalg.lstsq(Bd, -dd, rcond=None)[0].ravel()

    def solve(self, x0):
        U = solve_qp(self.P, (x0 @ self.TQS + self.tQS).ravel(),
                     lb=self.lb, ub=self.ub, solver=SOLVER)
        return U[:self.nu] if U is not None else np.zeros(self.nu)


def simulate(model, vx, R_curve, Ts, T_end, x0, controller):
    """실제 차량(연속시간)을 Ts 간격으로 적분. 조향은 물리적으로 항상 한계에서 잘린다."""
    Ac, Bc, dc = model.get_lti(vx, vx / R_curve)
    x = x0.copy(); hist = []
    for _ in np.arange(0.0, T_end, Ts):
        u = np.clip(controller(x), -model.delta_max, model.delta_max)
        hist.append(np.r_[x, u])
        x = solve_ivp(lambda tt, xx: Ac @ xx + Bc @ u + dc[:, 0], (0, Ts), x).y[:, -1]
    h = np.array(hist)
    return np.arange(len(h))*Ts, h[:, :4], h[:, 4]


def score(t, X, U, model):
    out = np.where(np.abs(X[:, 0]) > 0.3)[0]
    ts = t[-1] if len(out) == 0 or out[-1] == len(t)-1 else t[out[-1]+1]
    sat = float(np.mean(np.abs(U) >= model.delta_max - 1e-6)*100)
    return float(ts), float(np.abs(X[:, 0]).max()), sat, float(X[-1, 0])


# =========================================================== #
#  Part 1. 문제 설정
# =========================================================== #
print("=" * 78)
vx, Rc, Ts, T_end = 20.0, 50.0, 0.01, 5.0
Q = np.diag([1.0, 0.1, 0.1, 1.0]); Rw = np.diag([0.1])
x0 = np.array([8.0, 0.0, 0.30, 0.0])       # 8m 벗어나고 17도 틀어진 최악의 초기조건

print(f"[Part 1] {SEDAN.name}  |  {vx:.0f} m/s ({vx*3.6:.0f} km/h), 반경 {Rc:.0f}m 코너")
print(f"         횡가속도 {vx**2/Rc/9.81:.2f} g,  조향 한계 ±{SEDAN.delta_max*57.3:.0f}도")
print(f"         초기 오차: 횡방향 {x0[0]:.0f} m, 방위각 +{x0[2]*57.3:.0f}도 (더 벌어지는 방향)")
print(f"         제어주기 {Ts*1000:.0f} ms")
mpc30 = MPC(SEDAN, vx, vx/Rc, 30, Ts, Q, Rw)
print(f"         무제약 LQR 이 처음 요구하는 조향각: "
      f"{(-mpc30.Klqr @ x0 + mpc30.u_ff)[0]*57.3:.0f}도  <- 낼 수 없는 값")
summary["part1"] = dict(vx=vx, R=Rc, lat_g=round(vx**2/Rc/9.81, 3), Ts=Ts,
                        x0=x0.tolist(), delta_max_deg=round(SEDAN.delta_max*57.3, 1))


# =========================================================== #
#  Part 2. MPC vs LQR
# =========================================================== #
print("-" * 78)
print("[Part 2] 조향 한계가 걸리는 상황에서 두 제어기 비교")
print(f"{'제어기':<24}{'최대 |ey|':>13}{'최종 ey':>11}{'조향포화':>10}  결과")

K, uff = mpc30.Klqr, mpc30.u_ff
controllers = {
    "LQR (한계를 모름)": lambda x: -K @ x + uff,
    "MPC (N=30)":       mpc30.solve,
}
part2, res2 = {}, {}
for name, ctrl in controllers.items():
    t, X, U = simulate(SEDAN, vx, Rc, Ts, T_end, x0, ctrl)
    ts, emax, sat, efin = score(t, X, U, SEDAN)
    res2[name] = (t, X, U)
    diverged = emax > 50.0
    part2[name] = dict(max_ey=round(emax, 2), final_ey=round(efin, 3),
                       sat_pct=round(sat, 1), diverged=bool(diverged))
    print(f"{name:<24}{emax:>13.2f}{efin:>11.2f}{sat:>9.0f}%  "
          f"{'발산' if diverged else '복귀 성공'}")

print("\n         LQR 이 발산한다. 소프트웨어에서 자르든 하드웨어가 알아서 잘리든 결과는 같다.")
print("         LQR 은 '요구한 만큼 조향이 된다'고 믿고 이득을 계산했는데, 실제로는")
print("         한계에서 잘려 훨씬 약한 입력만 들어간다. 오차가 커지고, 커진 오차가")
print("         더 큰 요구를 낳고, 그 요구는 또 잘린다. 되돌아올 길이 없다.")
print("         MPC 는 한계를 알고 있으니 처음부터 '실제로 낼 수 있는 계획'만 세운다.")
print("         조향포화 17% -- 필요한 만큼만 한계에 붙였다 뗀다.")
summary["part2"] = part2


# =========================================================== #
#  Part 3. 예측구간 N 과 종단비용
# =========================================================== #
print("-" * 78)
print("[Part 3] 얼마나 멀리 봐야 하나 (Ts=10ms 이므로 N=30 은 0.3초 앞)")
print(f"{'N':>5}{'예측구간':>10}{'종단비용 O':>13}{'종단비용 X':>13}{'QP 1회':>11}{'주기대비':>10}")

part3, res3 = {}, {}
for N in [5, 10, 20, 30, 50]:
    row = {}
    for term in (True, False):
        mpc = MPC(SEDAN, vx, vx/Rc, N, Ts, Q, Rw, terminal=term)
        t, X, U = simulate(SEDAN, vx, Rc, Ts, T_end, x0, mpc.solve)
        row[term] = float(np.abs(X[:, 0]).max())
        if term:
            res3[N] = (t, X, U)
            t0 = time.perf_counter()
            for _ in range(200):
                mpc.solve(x0)
            ms = (time.perf_counter() - t0)/200*1000
    part3[f"N={N}"] = dict(horizon_s=round(N*Ts, 3), max_ey_terminal=round(row[True], 2),
                           max_ey_no_terminal=round(row[False], 2), qp_ms=round(ms, 3))
    f = lambda v: f"{v:>12.2f}" if v < 50 else f"{'발산':>12}"
    print(f"{N:>5}{N*Ts:>9.2f}s{f(row[True])}{f(row[False])}{ms:>10.3f}ms"
          f"{ms/(Ts*1000)*100:>9.1f}%")

print("\n         두 가지가 보인다.")
print("         (1) N 이 짧으면 실패한다. 0.05~0.2초는 20m/s 로 달리는 차에게 너무 가깝다.")
print("         (2) 종단비용이 오히려 해가 된다. 종단비용은 '예측구간 이후는 무제약 LQR")
print("             로 처리한다'는 가정인데, 조향이 한계에 걸린 상황에서 그 가정은 거짓이다.")
print("             '나중에 크게 꺾어 해결하면 된다'고 낙관해서 지금 덜 꺾는다.")
print("         N 이 충분히 길면 둘 다 같아진다. 종단 근처에서 이미 안정화됐기 때문이다.")
summary["part3_horizon"] = part3


# =========================================================== #
#  Part 4. 같은 MPC 를 FSAE 차량에
# =========================================================== #
print("-" * 78)
print("[Part 4] 제어기 코드는 그대로, 모델 파라미터만 갈아끼운다")
print(f"{'차량':<16}{'질량':>8}{'축거':>8}{'US그래디언트':>13}{'주행조건':>18}{'최대조향':>10}{'정착':>8}")

part4, res4 = {}, {}
vx_f, Rc_f = 15.0, 20.0
x0_f = np.array([2.0, 0.0, 0.30, 0.0])
for veh, vv, RR, xx0 in [(SEDAN, vx, Rc, x0), (FSAE, vx_f, Rc_f, x0_f)]:
    mpc = MPC(veh, vv, vv/RR, 30, Ts, Q, Rw)
    t, X, U = simulate(veh, vv, RR, Ts, T_end, xx0, mpc.solve)
    ts, emax, sat, efin = score(t, X, U, veh)
    Kus = veh.understeer_gradient()
    res4[veh.name] = (t, X, U, vv, RR, xx0)
    part4[veh.name] = dict(mass=veh.m, wheelbase=round(veh.lf+veh.lr, 2),
                           understeer_grad=round(Kus, 5), vx=vv, R=RR,
                           lat_g=round(vv**2/RR/9.81, 2), settle_s=round(ts, 3),
                           max_delta_deg=round(float(np.abs(U).max())*57.3, 1))
    print(f"{veh.name:<16}{veh.m:>7.0f}kg{veh.lf+veh.lr:>7.2f}m{Kus:>12.5f}"
          f"{f'{vv*3.6:.0f}km/h R={RR:.0f}m':>18}{np.abs(U).max()*57.3:>9.1f}도{ts:>7.2f}s")

print(f"\n         FSAE 는 {vx_f**2/Rc_f/9.81:.2f}g 로 도는데도 세단({vx**2/Rc/9.81:.2f}g)보다")
print(f"         훨씬 빨리 복귀한다. 질량 1/10, 관성 1/41 이라 그만큼 민첩하다.")
print(f"         제어기 코드는 한 줄도 안 바꿨다. 모델만 바꿨다. 그게 MPC 의 장점이다.")
summary["part4_vehicles"] = part4


# =========================================================== #
#  그림
# =========================================================== #
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
LBL2 = {"LQR (한계를 모름)": "LQR (unaware of limit)",
        "MPC (N=30)": "MPC (limit inside)"}

for name, c, ls in [("LQR (한계를 모름)", "tab:red", "--"),
                    ("MPC (N=30)", "tab:blue", "-")]:
    t, X, U = res2[name]
    axes[0, 0].plot(t, X[:, 0], color=c, ls=ls, lw=2, label=LBL2[name])
    axes[0, 1].plot(t, U*57.3, color=c, ls=ls, lw=2, label=LBL2[name])
axes[0, 0].axhline(0, color="k", ls="--", lw=1)
axes[0, 0].set(xlabel="t [s]", ylabel="lateral error ey [m]", xlim=(0, 5), ylim=(-30, 30),
               title="(a) LQR diverges once the steering saturates")
axes[0, 0].grid(alpha=.3); axes[0, 0].legend(fontsize=9)
for s in (1, -1):
    axes[0, 1].axhline(s*SEDAN.delta_max*57.3, color="k", ls=":", lw=1.5)
axes[0, 1].text(0.9, SEDAN.delta_max*57.3+2, "steering limit ±17°", fontsize=9)
axes[0, 1].set(xlabel="t [s]", ylabel="steering [deg]", xlim=(0, 3), ylim=(-40, 30),
               title="(b) steering command")
axes[0, 1].grid(alpha=.3); axes[0, 1].legend(fontsize=9)

for N, c in zip([5, 10, 20, 30, 50],
                ["tab:red", "tab:orange", "tab:olive", "tab:blue", "tab:purple"]):
    t, X, _ = res3[N]
    axes[0, 2].plot(t, X[:, 0], color=c, lw=1.8, label=f"N={N} ({N*Ts:.2f}s)")
axes[0, 2].axhline(0, color="k", ls="--", lw=1)
axes[0, 2].set(xlabel="t [s]", ylabel="lateral error ey [m]", xlim=(0, 5), ylim=(-30, 30),
               title="(c) too short a horizon fails")
axes[0, 2].grid(alpha=.3); axes[0, 2].legend(fontsize=8)

Ns = [5, 10, 20, 30, 50]
axes[1, 0].plot(Ns, [part3[f"N={N}"]["qp_ms"] for N in Ns], "o-", lw=2, color="tab:blue")
axes[1, 0].axhline(Ts*1000, color="r", ls="--", lw=1.5)
axes[1, 0].text(6, Ts*1000*1.15, f"control period {Ts*1000:.0f} ms", color="r", fontsize=9)
axes[1, 0].set(xlabel="prediction horizon N", ylabel="QP solve time [ms]", yscale="log",
               title="(d) cost of looking further")
axes[1, 0].grid(alpha=.3, which="both")

for (name, (t, X, U, vv, RR, xx0)), c in zip(res4.items(), ["tab:gray", "tab:blue"]):
    lab = "sedan 2980 kg" if "세단" in name else "FSAE 300 kg"
    axes[1, 1].plot(t, X[:, 0]/abs(xx0[0]), color=c, lw=2,
                    label=f"{lab}  {vv*3.6:.0f} km/h, R={RR:.0f} m")
    axes[1, 2].plot(t, U*57.3, color=c, lw=2, label=lab)
axes[1, 1].axhline(0, color="k", ls="--", lw=1)
axes[1, 1].set(xlabel="t [s]", ylabel="ey / ey(0)", xlim=(0, 3),
               title="(e) same controller, different car")
axes[1, 1].grid(alpha=.3); axes[1, 1].legend(fontsize=9)
axes[1, 2].set(xlabel="t [s]", ylabel="steering [deg]", xlim=(0, 3),
               title="(f) steering demand")
axes[1, 2].grid(alpha=.3); axes[1, 2].legend(fontsize=9)

fig.suptitle("MPC = re-solve the QP every step, use only the first move", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "mpc_study.png", dpi=120); plt.close(fig)

with open(OUT / "summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
print("=" * 78)
print(f"결과 저장: {OUT}")
