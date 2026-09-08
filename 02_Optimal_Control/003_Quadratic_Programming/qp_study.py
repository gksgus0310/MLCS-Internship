#==========================================#
# Title:  02-003 Quadratic Programming
#         Part 1. 제약이 하나씩 붙으면 해가 어떻게 밀려나는가
#         Part 2. 예제 코드의 active set 추측은 옳은가
#         Part 3. LQR 을 QP 로 다시 풀고, 드디어 |u| <= u_max 를 넣는다
#         Part 4. QP 한 번 푸는 데 얼마나 걸리나 (MPC 실시간성)
# Author: Hyun Han
# Date:   2026-09-08
#
# 002에서 남은 문제: "모터가 100까지밖에 못 내면?"
# LQR 은 등식(리카티)만 다루므로 부등식 제약을 넣을 수 없었다. QP 는 넣을 수 있다.
#==========================================#
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are, block_diag
from scipy.signal import cont2discrete
from qpsolvers import solve_qp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
summary = {}
SOLVER = "quadprog"

# 예제 코드와 같은 2차원 문제
# minimize  (1/2) x'Px + q'x
Pq = np.array([[1.0, 0.4],
               [0.4, 2.0]])
qq = np.array([0.6, 1.2])
Aeq = np.array([[2.0, -1.0]])            # 등식 제약 Ax = b
beq = np.array([1.0])
Gin = np.array([[-1.0, -2.0],            # 부등식 제약 Gx <= h (오각형)
                [-2.0, -1.0],
                [-1.0,  2.0],
                [ 1.0,  1.0],
                [ 2.0, -1.0]])
hin = np.array([1.0, 2.0, 6.0, 3.0, 3.0])
invP = np.linalg.inv(Pq)


# =========================================================== #
#  Part 1. 제약을 하나씩 붙여본다
# =========================================================== #
print("=" * 72)
print("[Part 1] 같은 목적함수에 제약만 바꿔가며 푼다")

x_unc = -invP @ qq                                        # 무제약: 해석해
x_eq  = solve_qp(Pq, qq, A=Aeq, b=beq, solver=SOLVER)     # 등식만
x_in  = solve_qp(Pq, qq, Gin, hin, solver=SOLVER)         # 부등식만
x_gen = solve_qp(Pq, qq, Gin, hin, Aeq, beq, solver=SOLVER)   # 둘 다

def J2(x):
    return float(0.5 * x @ Pq @ x + qq @ x)

part1 = {}
for name, x in [("무제약", x_unc), ("등식만", x_eq), ("부등식만", x_in), ("일반(둘다)", x_gen)]:
    part1[name] = dict(x=x.round(4).tolist(), cost=round(J2(x), 4))
    print(f"         {name:<12} x = [{x[0]:7.4f}, {x[1]:7.4f}]   비용 {J2(x):8.4f}")
print("         제약이 늘수록 비용은 반드시 커지거나 같다 (선택지가 줄어드니까)")
summary["part1"] = part1


# =========================================================== #
#  Part 2. 예제 코드의 active set 추측 검증
# =========================================================== #
print("-" * 72)
print("[Part 2] inequality_constrained_qp.py 는 이렇게 푼다:")
print("         '무제약 해에서 위반된 제약을 active 로 보고 등식으로 풀기'")

def naive_active_set(P, q, G, h):
    """예제 코드의 방식. 무제약 해에서 위반된 제약만 등식으로 걸고 한 번 푼다."""
    iP = np.linalg.inv(P)
    a = G @ (-iP @ q) > h
    if a.sum() == 0:
        return -iP @ q
    u = -np.linalg.solve(G[a] @ iP @ G[a].T, G[a] @ iP @ q + h[a])
    return -iP @ (G[a].T @ u + q)

x_naive = naive_active_set(Pq, qq, Gin, hin)
ok = np.allclose(x_naive, x_in, atol=1e-6)
print(f"         예제의 q 에서는: 추측 {x_naive.round(4)} vs 솔버 {x_in.round(4)} -> "
      f"{'일치' if ok else '불일치'}")

# q 를 무작위로 바꿔가며 이 방식이 항상 맞는지 확인
rng = np.random.default_rng(0)
n_try, n_fail, fails = 0, 0, []
for _ in range(3000):
    q_r = rng.uniform(-6, 6, 2)
    x_t = solve_qp(Pq, q_r, Gin, hin, solver=SOLVER)
    if x_t is None:
        continue
    try:
        x_n = naive_active_set(Pq, q_r, Gin, hin)
    except np.linalg.LinAlgError:
        continue
    n_try += 1
    if not np.allclose(x_n, x_t, atol=1e-5):
        n_fail += 1
        if len(fails) < 3:
            fails.append((q_r, x_n, x_t))

print(f"         무작위 q {n_try}개로 검증 -> {n_fail}개 실패 ({100*n_fail/n_try:.1f}%)")
for q_r, x_n, x_t in fails:
    print(f"           q={q_r.round(2)}  추측={x_n.round(3)}  정답={x_t.round(3)}")
print("         한 번의 추측으로는 안 된다. 진짜 active set 은 반복으로 찾아야 하고,")
print("         그 일을 solve_qp 같은 솔버가 해준다.")
summary["part2"] = dict(matches_example_q=bool(ok), tested=n_try, failed=n_fail,
                        fail_rate_pct=round(100 * n_fail / n_try, 2))


# =========================================================== #
#  Part 3. LQR 을 QP 로 -- 그리고 입력 제약을 넣는다
# =========================================================== #
print("-" * 72)
# 001, 002 와 같은 시스템
A = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
              [-3., 2., -2., 1.], [1., -1., .5, -1.]])
B = np.array([[0., 0.], [0., 0.], [1., 0.], [0., .5]])
n, m = A.shape[0], B.shape[1]
Ts, N = 0.05, 60                       # 0.05초 * 60 = 3초
Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)
Q = np.diag([1e2, 1e2, 1e0, 1e0])
R = 1e-2 * np.eye(m)
x0 = np.array([0.5, 1.0, 0.0, 0.0])

Pinf = solve_discrete_are(Ad, Bd, Q, R)          # 종단비용 = 무한구간 해
Klqr = np.linalg.solve(R + Bd.T @ Pinf @ Bd, Bd.T @ Pinf @ Ad)

# 예측행렬:  X = Sx x0 + Su U,   U = [u_0; ...; u_{N-1}]
Sx = np.zeros((n * (N + 1), n))
Su = np.zeros((n * (N + 1), m * N))
Sx[:n] = np.eye(n)
for k in range(N):
    Sx[n*(k+1):n*(k+2)] = Ad @ Sx[n*k:n*(k+1)]
    Su[n*(k+1):n*(k+2), :] = Ad @ Su[n*k:n*(k+1), :]
    Su[n*(k+1):n*(k+2), m*k:m*(k+1)] = Bd

Qbar = block_diag(*([Q] * N), Pinf)              # 마지막만 종단비용
Rbar = block_diag(*([R] * N))

# J(U) = (1/2)U' [2(Su'Qbar Su + Rbar)] U + [2 Su'Qbar Sx x0]' U + const
P_qp = 2 * (Su.T @ Qbar @ Su + Rbar)
P_qp = 0.5 * (P_qp + P_qp.T)                     # 대칭화 (수치오차 제거)
q_qp = 2 * Su.T @ Qbar @ Sx @ x0
const = float(x0 @ Sx.T @ Qbar @ Sx @ x0)

def true_cost(U):
    """실제 비용 J = sum(x'Qx + u'Ru) + x_N'P x_N"""
    return float(0.5 * U @ P_qp @ U + q_qp @ U + const)

def rollout(U):
    X = (Sx @ x0 + Su @ U).reshape(N + 1, n)
    return X, U.reshape(N, m)

UMAX = 20.0                                       # 모터가 낼 수 있는 최대 입력
VMAX = 1.5                                        # 속도 상태 |v1|, |v2| 한계
print(f"[Part 3] LQR 을 QP 로 정식화. 변수 {m*N}개 (u_0..u_{N-1}), 예측구간 {N*Ts:.1f}초")
print(f"         제약 두 가지: 입력 |u| <= {UMAX:g}  /  속도 |v| <= {VMAX:g}")

# 속도 상태에 대한 부등식: -VMAX <= v_k <= VMAX,  k=1..N
Csel = np.zeros((2, n)); Csel[0, 2] = 1.0; Csel[1, 3] = 1.0     # v1, v2 만 뽑는 행렬
Cbar = block_diag(*([np.zeros((2, n))] + [Csel] * N))           # k=0 은 초기상태라 제외
G_st = np.vstack([Cbar @ Su, -Cbar @ Su])
h_st = np.concatenate([VMAX - Cbar @ Sx @ x0, VMAX + Cbar @ Sx @ x0])

def viol(U):
    """입력/상태 제약 위반량"""
    X, Um = rollout(U)
    return float(max(0.0, np.abs(Um).max() - UMAX)), float(max(0.0, np.abs(X[:, 2:]).max() - VMAX))

# (a) 제약 없는 QP -> 유한구간 LQR 과 같다
U_free = solve_qp(P_qp, q_qp, solver=SOLVER)

# (b) LQR 로 풀고 입력만 한계에서 자르기 (흔히 하는 임시방편)
U_clip = np.zeros(m * N)
x = x0.copy()
for k in range(N):
    u = np.clip(-Klqr @ x, -UMAX, UMAX)
    U_clip[m*k:m*(k+1)] = u
    x = Ad @ x + Bd @ u

# (c) 입력 제약만 QP 로
U_qp_u = solve_qp(P_qp, q_qp, lb=-UMAX*np.ones(m*N), ub=UMAX*np.ones(m*N), solver=SOLVER)

# (d) 입력 + 상태 제약을 모두 QP 로
U_qp_all = solve_qp(P_qp, q_qp, G_st, h_st,
                    lb=-UMAX*np.ones(m*N), ub=UMAX*np.ones(m*N), solver=SOLVER)

part3 = {}
print(f"\n{'방법':<30}{'max|u|':>9}{'max|v|':>9}{'비용 J':>11}  제약")
J_best = true_cost(U_qp_all)
for name, U in [("(a) 제약 없는 QP (= LQR)", U_free),
                ("(b) LQR 후 입력 잘라내기", U_clip),
                ("(c) QP, 입력 제약만", U_qp_u),
                ("(d) QP, 입력+상태 제약", U_qp_all)]:
    X, Um = rollout(U)
    vu, vs = viol(U)
    umx, vmx, Jv = float(np.abs(Um).max()), float(np.abs(X[:, 2:]).max()), true_cost(U)
    tags = []
    if vu > 1e-6: tags.append("입력위반")
    if vs > 1e-6: tags.append("속도위반")
    part3[name] = dict(max_u=round(umx, 2), max_v=round(vmx, 3), J=round(Jv, 2),
                       input_violation=round(vu, 4), state_violation=round(vs, 4))
    print(f"{name:<30}{umx:>9.1f}{vmx:>9.2f}{Jv:>11.1f}  "
          f"{'  '.join(tags) if tags else 'OK'}")

print()
print("         (b) 자르기는 입력 한계는 지킨다. 그런데 속도는 1.5 를 훌쩍 넘는다.")
print("         자르기로는 '속도를 넘지 마라'를 표현할 방법이 아예 없다.")
print("         입력을 자른다고 상태가 어디로 갈지 통제되는 게 아니기 때문이다.")
print("         (d) 만 두 제약을 다 지킨다. QP 는 미래 궤적 전체를 보고 계획하니까.")
print()
print("         이것이 QP 가 LQR 보다 나은 지점이다. 더 좋은 답을 주는 게 아니라,")
print("         LQR 이 애초에 표현할 수 없는 요구를 표현할 수 있다.")
summary["part3"] = part3
summary["part3_setup"] = dict(N=N, Ts=Ts, umax=UMAX, vmax=VMAX, n_vars=m * N)


# =========================================================== #
#  Part 4. QP 한 번에 얼마나 걸리나
# =========================================================== #
print("-" * 72)
part4 = {}
for Nh in [10, 20, 40, 60, 100]:
    Sx_ = np.zeros((n * (Nh + 1), n)); Su_ = np.zeros((n * (Nh + 1), m * Nh))
    Sx_[:n] = np.eye(n)
    for k in range(Nh):
        Sx_[n*(k+1):n*(k+2)] = Ad @ Sx_[n*k:n*(k+1)]
        Su_[n*(k+1):n*(k+2), :] = Ad @ Su_[n*k:n*(k+1), :]
        Su_[n*(k+1):n*(k+2), m*k:m*(k+1)] = Bd
    Qb = block_diag(*([Q] * Nh), Pinf); Rb = block_diag(*([R] * Nh))
    Pn = 2 * (Su_.T @ Qb @ Su_ + Rb); Pn = 0.5 * (Pn + Pn.T)
    qn = 2 * Su_.T @ Qb @ Sx_ @ x0
    lb_, ub_ = -UMAX * np.ones(m * Nh), UMAX * np.ones(m * Nh)
    t0 = time.perf_counter()
    for _ in range(50):
        solve_qp(Pn, qn, lb=lb_, ub=ub_, solver=SOLVER)
    dt = (time.perf_counter() - t0) / 50 * 1000
    part4[f"N={Nh}"] = dict(n_vars=m * Nh, ms=round(dt, 3))
    print(f"[Part 4] N={Nh:>3} (변수 {m*Nh:>3}개)  QP 1회 {dt:7.3f} ms   "
          f"제어주기 {Ts*1000:.0f}ms 대비 {dt/(Ts*1000)*100:5.1f}%")
print("         MPC 는 매 스텝 이 QP 를 푼다. 제어주기 안에 끝나야 실시간 제어가 된다.")
summary["part4_timing"] = part4


# =========================================================== #
#  그림
# =========================================================== #
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

# --- 위: Part 1 의 네 가지 QP ---
gx, gy = np.meshgrid(np.linspace(-3, 3, 201), np.linspace(-3, 3, 201))
cost = 0.5 * (Pq[0, 0]*gx**2 + 2*Pq[0, 1]*gx*gy + Pq[1, 1]*gy**2) + qq[0]*gx + qq[1]*gy
poly_x = [1.0, -1.0, -2.0, 0.0, 2.0, 1.0]
poly_y = [-1.0, 0.0, 2.0, 3.0, 1.0, -1.0]

for ax, (title, x_s, eq, ineq) in zip(
        axes[0],
        [("(a) unconstrained", x_unc, False, False),
         ("(b) equality  Ax = b", x_eq, True, False),
         ("(c) inequality  Gx <= h", x_in, False, True)]):
    ax.contourf(gx, gy, cost, levels=18, cmap="viridis")
    if eq:
        ax.plot([-1.0, 2.0], [-3.0, 3.0], "w", lw=2, alpha=.8)
    if ineq:
        ax.plot(poly_x, poly_y, "w", lw=2, alpha=.8)
    ax.plot(*x_unc, "o", ms=7, mfc="none", mec="w", mew=1.5)
    ax.plot(*x_s, "r*", ms=18)
    ax.set(title=title, xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2")

# --- 아래 왼쪽: Part 1 일반 QP ---
ax = axes[1, 0]
ax.contourf(gx, gy, cost, levels=18, cmap="viridis")
ax.plot([-1.0, 2.0], [-3.0, 3.0], "w", lw=2, alpha=.8)
ax.plot(poly_x, poly_y, "w", lw=2, alpha=.8)
ax.plot(*x_unc, "o", ms=7, mfc="none", mec="w", mew=1.5, label="unconstrained")
ax.plot(*x_gen, "r*", ms=18, label="solution")
ax.set(title="(d) general  Ax=b  and  Gx<=h", xlim=(-3, 3), ylim=(-3, 3),
       xlabel="x1", ylabel="x2")
ax.legend(fontsize=8, loc="lower left")

# --- 아래 가운데: Part 3 입력 ---
ax = axes[1, 1]
tgrid = np.arange(N) * Ts
for name, U, c, ls in [("(a) no limits (=LQR)", U_free, "tab:gray", "-"),
                       ("(b) LQR then clip", U_clip, "tab:red", "--"),
                       ("(d) QP: input + state", U_qp_all, "tab:blue", "-")]:
    ax.plot(tgrid, U.reshape(N, m)[:, 1], color=c, ls=ls, lw=2, label=name)
ax.axhline(UMAX, color="k", ls=":", lw=1.5)
ax.axhline(-UMAX, color="k", ls=":", lw=1.5)
ax.text(0.55, UMAX + 4, f"motor limit |u| = {UMAX:g}", fontsize=9)
ax.set(xlabel="t [s]", ylabel="control input u2", title="(e) input: limit as a constraint",
       xlim=(0, 1.5), ylim=(-90, 45))
ax.grid(alpha=.3); ax.legend(fontsize=9)

# --- 아래 오른쪽: Part 3 상태 ---
ax = axes[1, 2]
for name, U, c, ls in [("(a) no limits (=LQR)", U_free, "tab:gray", "-"),
                       ("(b) LQR then clip", U_clip, "tab:red", "--"),
                       ("(d) QP: input + state", U_qp_all, "tab:blue", "-")]:
    X, _ = rollout(U)
    ax.plot(np.arange(N + 1) * Ts, X[:, 3], color=c, ls=ls, lw=2, label=name)
ax.axhline(-VMAX, color="k", ls=":", lw=1.5)
ax.axhline(VMAX, color="k", ls=":", lw=1.5)
ax.text(1.6, -VMAX + .12, f"speed limit |v| = {VMAX:g}", fontsize=9)
ax.set(xlabel="t [s]", ylabel="state v2 (speed)", xlim=(0, 3), ylim=(-3, .5),
       title="(f) only QP respects the STATE limit")
ax.grid(alpha=.3); ax.legend(fontsize=9)

fig.suptitle("QP: this is where 'the motor can only do so much' finally fits in", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "qp_study.png", dpi=120)
plt.close(fig)

with open(OUT / "summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
print("=" * 72)
print(f"결과 저장: {OUT}")
