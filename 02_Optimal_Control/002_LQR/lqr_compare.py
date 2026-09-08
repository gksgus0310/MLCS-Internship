#==========================================#
# Title:  02-002 LQR (Linear Quadratic Regulator)
#         Part 1. LQR 이 극점을 '알아서' 정해준다
#         Part 2. R 하나로 속도-제어량 저울질을 조절한다
#         Part 3. 극점배치와 정면 비교 - 최적이라는 게 무슨 뜻인가
#         Part 4. 연속 LQR vs 이산 LQR
# Author: Hyun Han
# Date:   2026-09-08
#
# 001에서 남은 질문: "극점을 무슨 기준으로 고를 것인가?"
# LQR 의 답: 고르지 마라. 비용함수를 정하면 극점은 결과로 따라온다.
#==========================================#
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space, solve_continuous_are, solve_discrete_are, solve_lyapunov
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
summary = {}

# 001 과 완전히 같은 시스템
A = np.array([
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-3.0,  2.0, -2.0,  1.0],
    [ 1.0, -1.0,  0.5, -1.0],
])
B = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.5],
])
n, m = A.shape[0], B.shape[1]      # 상태 4개, 입력 2개

# 조정 문제로 본다: x0 에서 시작해 원점으로. u = -Kx
# (001 의 추종 문제와 같은 것을 좌표만 옮긴 형태)
x0 = np.array([0.5, 1.0, 0.0, 0.0])
T_END = 5.0
t_eval = np.linspace(0.0, T_END, 1001)

Q = np.diag([1e2, 1e2, 1e0, 1e0])       # 예제 코드와 동일
R_BASE = np.eye(m)


# ----------------------------------------------------------- #
def lqr(A, B, Q, R):
    """연속시간 LQR. 리카티 방정식을 풀어 최적 이득 K 를 얻는다.

        A'P + PA - PBR^-1B'P + Q = 0        (CARE)
        K = R^-1 B' P
    """
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P), P


def cost_of_gain(K, Q, R):
    """이득 K 를 썼을 때의 비용 J = x0' P x0.

    닫힌루프 Ac = A - BK 에 대해 리아푸노프 방정식
        Ac'P + P Ac + (Q + K'RK) = 0
    을 풀면 J = integral(x'Qx + u'Ru) dt 를 정확히 얻는다.
    LQR 의 K 가 이 J 를 최소화한다는 것이 '최적'의 정의다.
    """
    Ac = A - B @ K
    if np.max(np.linalg.eigvals(Ac).real) >= 0:
        return np.inf
    P = solve_lyapunov(Ac.T, -(Q + K.T @ R @ K))
    return float(x0 @ P @ x0)


def simulate(K):
    sol = solve_ivp(lambda t, x: (A - B @ K) @ x, (0.0, T_END), x0, t_eval=t_eval)
    return sol.t, sol.y, -K @ sol.y            # u = -Kx


def metrics(t, y, u, tol=0.02):
    """정착시간(위치 상태 기준)과 최대 제어입력."""
    ts = 0.0
    for k in (0, 1):
        band = tol * abs(x0[k])
        out = np.where(np.abs(y[k]) > band)[0]
        ts = max(ts, t[-1] if len(out) == 0 or out[-1] == len(t) - 1 else t[out[-1] + 1])
    return ts, float(np.abs(u).max())


def eigenstructure_K(A, B, eigs):
    """001 에서 쓴 극점배치. 비교용."""
    P = np.hstack([null_space(np.hstack([e * np.eye(n) - A, B])) for e in eigs])
    return np.real(P[n:, :] @ np.linalg.inv(P[:n, :]))


# =========================================================== #
#  Part 1. LQR 은 극점을 결과로 내놓는다
# =========================================================== #
print("=" * 74)
R0 = 1e-2 * np.eye(m)                          # 예제 코드와 동일
K0, P0 = lqr(A, B, Q, R0)
eig0 = np.sort_complex(np.linalg.eigvals(A - B @ K0))

print("[Part 1] Q = diag(100, 100, 1, 1),  R = 0.01 I")
for row in K0:
    print("           K = " if row is K0[0] else "               ",
          np.array2string(row, precision=2, suppress_small=True))
print(f"         닫힌루프 극점: {np.round(eig0, 2)}")
print(f"         -> 극점을 지정한 적이 없다. 비용함수를 정했더니 나온 값이다.")
print(f"         비용 J = {cost_of_gain(K0, Q, R0):.3f}")
summary["part1"] = dict(Q=np.diag(Q).tolist(), R=float(R0[0, 0]),
                        K=K0.round(3).tolist(),
                        poles=[[float(e.real), float(e.imag)] for e in eig0],
                        J=round(cost_of_gain(K0, Q, R0), 4))


# =========================================================== #
#  Part 2. R 하나로 저울질을 조절한다
# =========================================================== #
print("-" * 74)
print("[Part 2] R 을 키우면 제어입력을 아끼고, 줄이면 빨라진다")
print(f"{'R':>10}{'정착시간':>12}{'max|u|':>12}{'비용 J':>12}{'느린 극점':>14}")

SHOW_RHOS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
rhos = np.unique(np.concatenate([np.logspace(-4, 1, 60), SHOW_RHOS]))
sweep = {"rho": [], "ts": [], "umax": [], "J": []}

for rho in rhos:
    K, _ = lqr(A, B, Q, rho * R_BASE)
    t, y, u = simulate(K)
    ts, umax = metrics(t, y, u)
    J = cost_of_gain(K, Q, R0)                 # 비교를 위해 항상 같은 잣대(R0)로 평가
    sweep["rho"].append(float(rho)); sweep["ts"].append(float(ts))
    sweep["umax"].append(float(umax)); sweep["J"].append(float(J))
    if any(abs(rho - s) < 1e-12 for s in SHOW_RHOS):
        slow = max(np.linalg.eigvals(A - B @ K).real)
        star = "  <- 예제 코드의 R" if abs(rho - 1e-2) < 1e-12 else ""
        print(f"{rho:>10.4g}{ts:>12.2f}{umax:>12.1f}{J:>12.2f}{slow:>14.2f}{star}")

summary["part2_sweep"] = sweep


# =========================================================== #
#  Part 3. 극점배치와 정면 비교
# =========================================================== #
print("-" * 74)
print("[Part 3] 모든 제어기를 같은 잣대로 평가한다")
print("         비용 J = integral(x'Qx + u'Ru)dt,  Q=diag(100,100,1,1), R=0.01I")
print(f"{'방법':<24}{'정착시간':>10}{'max|u|':>11}{'비용 J':>11}{'최적 대비':>11}")

POLE_SETS = {"극점배치 (-2,-1.5)": [-2.0, -1.5],
             "극점배치 (-10,-8)": [-10.0, -8.0],
             "극점배치 (-25,-20)": [-25.0, -20.0]}
part3 = {}
pp_points = []
J_opt = cost_of_gain(K0, Q, R0)

t, y, u = simulate(K0)
ts0, umax0 = metrics(t, y, u)
print(f"{'LQR (R=0.01I)':<24}{ts0:>10.2f}{umax0:>11.1f}{J_opt:>11.2f}{'기준':>11}")
part3["LQR (R=0.01I)"] = dict(ts=round(ts0, 3), umax=round(umax0, 2), J=round(J_opt, 3))

for name, eigs in POLE_SETS.items():
    K = eigenstructure_K(A, B, eigs)
    t, y, u = simulate(K)
    ts, umax = metrics(t, y, u)
    J = cost_of_gain(K, Q, R0)
    part3[name] = dict(ts=round(ts, 3), umax=round(umax, 2), J=round(J, 3),
                       vs_optimal_pct=round((J / J_opt - 1) * 100, 1))
    pp_points.append((ts, umax, J, name))
    print(f"{name:<24}{ts:>10.2f}{umax:>11.1f}{J:>11.2f}{(J/J_opt-1)*100:>10.1f}%")

print()
print("  LQR 이 J 를 최소화한다는 것이 리카티 방정식이 보장하는 내용이다.")
print("  어떤 K 를 가져와도 26.6 보다 작아질 수 없다.")
print()
print("  주의: 그렇다고 LQR 이 '가장 빠른' 제어기는 아니다.")
ts_fast = part3["극점배치 (-25,-20)"]["ts"]
print(f"  극점배치(-25,-20)은 {ts_fast:.2f}s 로 LQR({ts0:.2f}s)보다 빠르다.")
print("  대신 max|u| 를 999 까지 쓴다. J 는 그 낭비를 비용으로 계산해 벌점을 준다.")
print("  '최적'은 내가 정의한 비용함수에 대해서만 최적이다.")

summary["part3"] = part3


# =========================================================== #
#  Part 4. 연속 LQR vs 이산 LQR
# =========================================================== #
print("-" * 74)
part4 = {}
for Ts in [0.05, 0.1, 0.2]:
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)
    Pd = solve_discrete_are(Ad, Bd, Q, R0)
    Kd = np.linalg.solve(R0 + Bd.T @ Pd @ Bd, Bd.T @ Pd @ Ad)
    rho_d = np.abs(np.linalg.eigvals(Ad - Bd @ Kd)).max()
    part4[f"Ts={Ts}"] = dict(spectral_radius=round(float(rho_d), 4), stable=bool(rho_d < 1))
    print(f"[Part 4] 이산 LQR  Ts={Ts:<5} 스펙트럴 반경 {rho_d:.3f}  "
          f"{'안정' if rho_d < 1 else '불안정'}")
print("         001 의 emulation 은 Ts=0.15 에서 발산했다. 이산 LQR 은 안 그런다.")
summary["part4_discrete"] = part4


# =========================================================== #
#  그림
# =========================================================== #
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# (a) R 에 따른 응답
for rho, c in zip([1e-4, 1e-2, 1e0, 1e1], ["tab:red", "tab:blue", "tab:orange", "tab:green"]):
    K, _ = lqr(A, B, Q, rho * R_BASE)
    t, y, _ = simulate(K)
    ax_a.plot(t, y[1], color=c, lw=2, label=f"R = {rho:g} I")
ax_a.axhline(0, color="k", ls="--", lw=1)
ax_a.set(xlabel="t [s]", ylabel="state p2", xlim=(0, 3),
         title="(a) one knob: larger R = gentler control")
ax_a.grid(alpha=.3); ax_a.legend(fontsize=9)

# (b) 속도 vs 제어량
ax_b.plot(sweep["umax"], sweep["ts"], "-", color="tab:blue", lw=2.5, label="LQR (R sweep)")
for ts_pp, umax_pp, _, name in pp_points:
    ax_b.plot(umax_pp, ts_pp, "X", ms=13, color="tab:red")
    ax_b.annotate(name.replace("극점배치 ", ""), (umax_pp, ts_pp),
                  textcoords="offset points", xytext=(8, 8), fontsize=9, color="tab:red")
ax_b.plot([], [], "X", ms=11, color="tab:red", label="pole placement (001)")
ax_b.set(xscale="log", xlabel="max |u|  (control effort)", ylabel="settling time [s]",
         title="(b) speed vs effort — settling time is NOT what LQR minimizes")
ax_b.grid(alpha=.3, which="both"); ax_b.legend(fontsize=9)

# (c) 극점 자취
for rho in np.logspace(-4, 1, 30):
    K, _ = lqr(A, B, Q, rho * R_BASE)
    e = np.linalg.eigvals(A - B @ K)
    ax_c.scatter(e.real, e.imag, s=16, c=[np.log10(rho)] * n, cmap="viridis", vmin=-4, vmax=1)
sc = ax_c.scatter([], [], c=[], cmap="viridis", vmin=-4, vmax=1)
plt.colorbar(sc, ax=ax_c, label="log10(R)")
ax_c.axvline(0, color="k", lw=1)
ax_c.set(xlabel="Re", ylabel="Im", title="(c) poles move as a result, not as an input")
ax_c.grid(alpha=.3)

# (d) 비용 J - 여기가 LQR 이 이기는 곳
names = ["LQR\n(R=0.01I)"] + [n.replace("극점배치 ", "poles\n") for n in POLE_SETS]
Js = [J_opt] + [part3[n]["J"] for n in POLE_SETS]
colors = ["tab:blue"] + ["tab:red"] * 3
bars = ax_d.bar(names, Js, color=colors, alpha=.85)
ax_d.axhline(J_opt, color="tab:blue", ls="--", lw=1.5)
ax_d.text(2.4, J_opt * 1.15, f"minimum J = {J_opt:.1f}\n(nothing can go below)",
          color="tab:blue", fontsize=9)
for b, J in zip(bars, Js):
    ax_d.text(b.get_x() + b.get_width() / 2, J + 3, f"{J:.1f}", ha="center", fontsize=10)
ax_d.set(ylabel="cost  J = ∫(x'Qx + u'Ru) dt",
         title="(d) same yardstick: LQR is provably minimal")
ax_d.grid(alpha=.3, axis="y")

fig.suptitle("LQR: define the cost, and the poles follow", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "lqr_tradeoff.png", dpi=120)
plt.close(fig)

with open(OUT / "summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
print("=" * 74)
print(f"결과 저장: {OUT}")
