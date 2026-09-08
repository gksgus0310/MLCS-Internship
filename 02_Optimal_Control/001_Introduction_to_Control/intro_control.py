#==========================================#
# Title:  02-001 Introduction to Control
#         Part 1. 가제어성 + 상태피드백(eigenstructure assignment) 극점 선택 비교
#         Part 2. 이산화 - 샘플링 주기가 성능을 어떻게 무너뜨리는가
#         Part 3. 비선형 시스템 - 이동로봇 자세 제어
# Author: Hyun Han
# Date:   2026-08-25
#
# 예제 코드(linear_systems.py / nonlinear_systems.py)를 기반으로,
# "극점을 어디에 둘 것인가"와 "이산화하면 무슨 일이 생기는가"를
# 직접 비교해 보는 것이 목적.
#==========================================#
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space, expm
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

summary = {}

# =========================================================== #
#  대상 시스템: 2자유도 질량-스프링-댐퍼 형태의 4상태 선형 시스템
#    상태 x = [p1, p2, v1, v2],  입력 u = [u1, u2]
# =========================================================== #
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
n, m = A.shape[0], B.shape[1]

STATE_LABELS = ["p1", "p2", "v1", "v2"]
x0   = np.zeros(n)
xref = np.array([0.5, 1.0, 0.0, 0.0])
T_END = 5.0


# ----------------------------------------------------------- #
def controllability(A, B):
    """가제어성 행렬 [B AB A^2B ... A^(n-1)B] 의 랭크.

    rank == n 이어야 상태피드백으로 극점을 마음대로 배치할 수 있다.
    (강의자료의 'if the system is completely controllable' 조건)
    """
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return C, np.linalg.matrix_rank(C)


def eigenstructure_K(A, B, eigs):
    """Eigenstructure assignment 로 상태피드백 이득 K 를 구한다.

    [ (λI - A)  B ] [ v ; Kv ] = 0  을 만족하는 null space 를 모으면
    (A - BK) v = λ v  가 되어 닫힌루프 고유값이 λ 가 된다.
    입력이 m개면 고유값 하나당 m개의 방향을 얻으므로,
    n = m * (고유값 개수) 가 되도록 고유값을 고른다.
    """
    n = A.shape[0]
    P = np.hstack([null_space(np.hstack([e * np.eye(n) - A, B])) for e in eigs])
    V, Q = P[:n, :], P[n:, :]
    return np.real(Q @ np.linalg.inv(V))


def simulate_continuous(K, t_eval):
    """연속시간 닫힌루프 시뮬레이션. 상태와 입력을 함께 돌려준다."""
    sol = solve_ivp(
        lambda t, x: A @ x + B @ K @ (xref - x),
        (0.0, t_eval[-1]), x0, t_eval=t_eval,
    )
    u = K @ (xref[:, None] - sol.y)          # u(t) = K (xref - x)
    return sol.t, sol.y, u


def settling_time(t, y, target, tol=0.02):
    """2% 정착시간. 마지막으로 허용대역을 벗어난 시점 이후를 정착으로 본다."""
    band = tol * max(abs(target), 1e-9)
    outside = np.where(np.abs(y - target) > band)[0]
    return t[-1] if len(outside) == 0 or outside[-1] == len(t) - 1 else t[outside[-1] + 1]


# =========================================================== #
#  Part 1. 극점을 어디에 둘 것인가
# =========================================================== #
print("=" * 70)
C, rank = controllability(A, B)
print(f"[Part 1] 가제어성 행렬 크기 {C.shape}, rank = {rank} (n = {n})")
print(f"         -> {'완전 가제어. 극점을 임의로 배치할 수 있다.' if rank == n else '가제어 아님!'}")
print(f"         개루프 고유값: {np.sort_complex(np.linalg.eigvals(A)).round(3)}")

POLE_SETS = {
    "slow  (-2, -1.5)":  [-2.0, -1.5],
    "given (-10, -8)":   [-10.0, -8.0],   # 예제 코드 원본
    "fast  (-25, -20)":  [-25.0, -20.0],
}

t_eval = np.linspace(0.0, T_END, 501)
part1 = {}
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)

for (name, eigs), color in zip(POLE_SETS.items(), ["tab:green", "tab:blue", "tab:red"]):
    K = eigenstructure_K(A, B, eigs)
    t, y, u = simulate_continuous(K, t_eval)

    ts = max(settling_time(t, y[k], xref[k]) for k in (0, 1))
    umax = np.abs(u).max()
    # 정상상태: (A-BK)x_ss + BK*xref = 0  ->  x_ss = -(A-BK)^-1 B K xref
    x_ss = -np.linalg.solve(A - B @ K, B @ K @ xref)
    ss_err = float(np.abs(x_ss[:2] - xref[:2]).max())
    part1[name] = dict(eigs=eigs, K=K.round(3).tolist(),
                       settling_time_s=round(float(ts), 3),
                       max_abs_u=round(float(umax), 2),
                       max_abs_K=round(float(np.abs(K).max()), 1),
                       steady_state_error=round(ss_err, 4))
    print(f"         {name:<18} 정착 {ts:5.2f}s | max|u| {umax:8.1f} | "
          f"max|K| {np.abs(K).max():7.1f} | 정상상태오차 {ss_err:.4f}")

    for k in (0, 1):                                  # 위치 상태 p1, p2
        axes[0, k].plot(t, y[k], color=color, label=name)
    for j in (0, 1):                                  # 제어입력 u1, u2
        axes[1, j].plot(t, u[j], color=color, label=name)

for k in (0, 1):
    axes[0, k].axhline(xref[k], color="k", ls="--", lw=1, label="reference" if k == 0 else None)
    axes[0, k].set_title(f"state {STATE_LABELS[k]}"); axes[0, k].grid(alpha=.3)
    axes[1, k].set_title(f"control input u{k+1}"); axes[1, k].grid(alpha=.3)
    axes[1, k].set_xlabel("t [s]")
axes[0, 0].legend(fontsize=9); axes[1, 0].set_yscale("symlog")
axes[1, 1].set_yscale("symlog")
fig.suptitle("Part 1. Pole placement: faster poles cost control effort", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "part1_pole_placement.png", dpi=120); plt.close(fig)
summary["part1_pole_placement"] = part1
summary["controllability_rank"] = int(rank)


# =========================================================== #
#  Part 2. 이산화 - 실제 구현은 컴퓨터에서 돌아간다
# =========================================================== #
print("-" * 70)
K = eigenstructure_K(A, B, [-10.0, -8.0])            # 연속시간 설계는 고정
t_c, y_c, _ = simulate_continuous(K, t_eval)         # 기준이 되는 연속시간 응답


def discrete_gain(Ad, Bd, Ts, method):
    """이산 제어기를 얻는 두 가지 방법.

    'emulation' : 연속시간 이득 K 를 그대로 가져다 쓴다 (가장 흔한 실수).
    'redesign'  : 연속 닫힌루프 천이행렬 expm((A-BK)Ts) 를 재현하도록
                  이산 이득을 다시 구한다 (예제 코드가 하는 것).
    """
    if method == "emulation":
        return K
    return np.linalg.pinv(Bd) @ (Ad - expm((A - B @ K) * Ts))


def simulate_discrete(Kd, Ad, Bd, Ts):
    td = np.arange(0.0, T_END, Ts)
    yd = [x0.copy()]
    for _ in td[1:]:
        yd.append(Ad @ yd[-1] + Bd @ Kd @ (xref - yd[-1]))
    return td, np.array(yd).T


TS_LIST = [0.05, 0.1, 0.15, 0.2]
COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
part2 = {}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax in axes[:2]:
    ax.plot(t_c, y_c[0], "k", lw=2, label="continuous (ideal)")
    ax.axhline(xref[0], color="gray", ls="--", lw=1)

for col, method in enumerate(["emulation", "redesign"]):
    for Ts, color in zip(TS_LIST, COLORS):
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)
        Kd = discrete_gain(Ad, Bd, Ts, method)
        rho = np.abs(np.linalg.eigvals(Ad - Bd @ Kd)).max()   # <1 이면 안정
        td, yd = simulate_discrete(Kd, Ad, Bd, Ts)

        part2.setdefault(method, {})[f"Ts={Ts}"] = dict(
            spectral_radius=round(float(rho), 3), stable=bool(rho < 1.0),
            final_error=round(float(np.abs(yd[0, -1] - xref[0])), 4))
        print(f"[Part 2] {method:<10} Ts={Ts:<5} rho={rho:6.3f}  "
              f"{'안정' if rho < 1 else '불안정 (발산)'}")

        axes[col].plot(td, yd[0], color=color, marker="o", ms=2.5, lw=1,
                       label=f"Ts={Ts}s (rho={rho:.2f})")

    axes[col].set(xlabel="t [s]", ylabel="state p1", ylim=(-0.6, 1.4),
                  title=f"(a) emulation: reuse continuous K" if method == "emulation"
                        else "(b) redesign: recompute discrete gain")
    axes[col].grid(alpha=.3); axes[col].legend(fontsize=8)

# 샘플링 주기에 따른 스펙트럴 반경
ts_fine = np.linspace(0.005, 0.25, 60)
for method, style in [("emulation", "-"), ("redesign", "--")]:
    rhos = []
    for Ts in ts_fine:
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)
        Kd = discrete_gain(Ad, Bd, Ts, method)
        rhos.append(np.abs(np.linalg.eigvals(Ad - Bd @ Kd)).max())
    axes[2].plot(ts_fine, rhos, style, lw=2, label=method)
axes[2].axhline(1.0, color="r", lw=1.5)
axes[2].text(0.13, 1.05, "unstable above this line", color="r", fontsize=9)
axes[2].set(xlabel="sampling time Ts [s]", ylabel="spectral radius",
            title="stability vs sampling time", ylim=(0, 2.2))
axes[2].grid(alpha=.3); axes[2].legend(fontsize=9)

fig.suptitle("Part 2. Discretization: the same controller can go unstable", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "part2_discretization.png", dpi=120); plt.close(fig)
summary["part2_discretization"] = part2


# =========================================================== #
#  Part 3. 비선형 시스템 - 이동로봇 자세 제어
#    극좌표 오차 e = [r, alpha, phi] 에 대한 비선형 피드백.
#    선형 시스템처럼 '극점 배치'가 아니라 안정성 논증으로 설계된다.
# =========================================================== #
print("-" * 70)
LAMBDA_ALPHA, LAMBDA_PHI = 1.0, 1.0
K_V, K_OMEGA = 2.0, 3.0


def f_robot(e, u):
    r, alpha, _ = e
    v, omega = u
    return np.array([-v * np.cos(alpha),
                     v * np.sin(alpha) / r - omega,
                     -v * np.sin(alpha) / r])


def k_robot(e):
    r, alpha, phi = e
    sinc_term = np.sin(alpha) / alpha if abs(alpha) > 1e-3 else 1.0
    return np.array([
        K_V * r * np.cos(alpha),
        K_V * (LAMBDA_ALPHA * alpha - LAMBDA_PHI * phi) * np.cos(alpha) * sinc_term / LAMBDA_ALPHA
        + K_OMEGA * alpha,
    ])


INITIAL_POSES = [
    (3.0, -np.pi / 2, np.pi / 4),      # 예제 코드 원본
    (3.0,  np.pi / 2, -np.pi / 4),
    (2.0, -np.pi / 3, np.pi / 2),
    (4.0,  np.pi / 4, np.pi / 3),
    (1.5, -np.pi / 4, -np.pi / 3),
    (3.5,  np.pi / 3, -np.pi / 6),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
t_eval_r = np.linspace(0.0, 5.0, 501)
part3 = {}

for i, e0 in enumerate(INITIAL_POSES):
    sol = solve_ivp(lambda t, e: f_robot(e, k_robot(e)), (0.0, 5.0),
                    np.array(e0), t_eval=t_eval_r)
    r, alpha, phi = sol.y
    # 극좌표 오차 -> 전역 좌표계에서의 로봇 위치/방향
    x = np.array([-r * np.cos(phi), -r * np.sin(phi), alpha + phi])

    part3[f"pose{i+1}"] = dict(e0=[round(v, 3) for v in e0],
                               final_r=round(float(r[-1]), 4))
    axes[0].plot(sol.t, r, lw=1.5, label=f"r0={e0[0]}")
    axes[1].plot(x[0], x[1], lw=1.5)
    axes[1].plot(x[0, 0], x[1, 0], "o", ms=5)
    axes[1].quiver(x[0, ::40], x[1, ::40], np.cos(x[2, ::40]), np.sin(x[2, ::40]),
                   scale=18.0, alpha=.35, width=.004)

print(f"[Part 3] 초기자세 {len(INITIAL_POSES)}개 모두 시뮬레이션")
print(f"         최종 거리오차 r: 최대 {max(v['final_r'] for v in part3.values()):.4f} "
      f"(0에 수렴하면 성공)")

axes[0].axhline(0, color="k", ls="--", lw=1)
axes[0].set(xlabel="t [s]", ylabel="distance error r [m]", title="error convergence")
axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
axes[1].plot(0, 0, "k*", ms=15, label="goal")
axes[1].set(xlabel="x [m]", ylabel="y [m]", title="robot trajectories")
axes[1].axis("equal"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=9)
fig.suptitle("Part 3. Nonlinear pose control of a mobile robot", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "part3_mobile_robot.png", dpi=120); plt.close(fig)
summary["part3_mobile_robot"] = part3

# ----------------------------------------------------------- #
with open(OUT / "summary.json", "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)
print("=" * 70)
print(f"결과 저장: {OUT}")
