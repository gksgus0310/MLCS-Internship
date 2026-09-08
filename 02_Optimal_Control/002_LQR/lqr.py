#==========================================#
# Title:  02-002 Optimal Control of Linear Systems
#         과제: lqr.py 가 쓰던 scipy 의 solve_continuous_are / solve_discrete_are 를
#               강의자료의 Iterative solution 식으로 직접 구현하기
# Author: Hyun Han
# Date:   2026-09-09
#
# 강의자료의 반복 알고리즘 (연속시간):
#     K <- 0
#     repeat until P converges
#         P <- vec^-1( -(I (x) (A-BK)^T + (A-BK)^T (x) I)^-1  vec(Q + K^T R K) )
#         K <- R^-1 B^T P
#     end repeat
#
# 안쪽 한 줄은 리아푸노프 방정식  (A-BK)^T P + P(A-BK) + (Q + K^T R K) = 0  을
# vec 연산으로 선형계로 바꿔 푼 것이다.  vec(AXB) = (B^T (x) A) vec(X) 를 쓰면
#     vec((A-BK)^T P) = (I (x) (A-BK)^T) vec(P)
#     vec(P (A-BK))   = ((A-BK)^T (x) I) vec(P)
# 이므로 두 항을 더해 vec(P) 에 대해 풀면 위 식이 된다.
#==========================================#
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are   # 검증용으로만 사용
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- #
#  vec 연산 (열 우선). numpy 는 기본이 행 우선이라 order='F' 를 명시한다.
# ---------------------------------------------------------------- #
def vec(M):
    return M.reshape(-1, order="F")


def unvec(v, n):
    return v.reshape((n, n), order="F")


# ---------------------------------------------------------------- #
#  연속시간 CARE 를 반복법으로 (강의자료 식 그대로)
# ---------------------------------------------------------------- #
def care_iterative(A, B, Q, R, tol=1e-12, max_iter=200, verbose=False):
    """A'P + PA - PBR^-1B'P + Q = 0 을 반복법으로 푼다.

    K=0 에서 시작하려면 A 가 이미 안정해야 한다 (아래 주석 참고).
    """
    n = A.shape[0]
    In = np.eye(n)
    K = np.zeros((B.shape[1], n))          # K <- 0
    residuals = []

    for it in range(max_iter):
        M = A - B @ K                       # 현재 정책의 닫힌루프 행렬
        N = Q + K.T @ R @ K                 # 현재 정책의 비용 가중치

        # 리아푸노프: M'P + PM + N = 0  ->  (I(x)M' + M'(x)I) vec(P) = -vec(N)
        L = np.kron(In, M.T) + np.kron(M.T, In)
        P = unvec(-np.linalg.solve(L, vec(N)), n)
        P = 0.5 * (P + P.T)                 # 대칭화 (수치오차 정리)

        K = np.linalg.solve(R, B.T @ P)     # K <- R^-1 B' P

        # 원래 리카티 방정식에 넣어 잔차를 본다
        res = np.abs(A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q).max()
        residuals.append(res)
        if verbose:
            print(f"           iter {it+1:>3}  residual {res:.3e}")
        if res < tol:
            break

    return P, K, residuals


# ---------------------------------------------------------------- #
#  이산시간 DARE 도 같은 방식으로
#     M'PM - P + N = 0  ->  (M'(x)M' - I) vec(P) = -vec(N)
# ---------------------------------------------------------------- #
def dare_iterative(A, B, Q, R, tol=1e-12, max_iter=200):
    n = A.shape[0]
    I2 = np.eye(n * n)
    K = np.zeros((B.shape[1], n))
    residuals = []

    for _ in range(max_iter):
        M = A - B @ K
        N = Q + K.T @ R @ K
        P = unvec(np.linalg.solve(I2 - np.kron(M.T, M.T), vec(N)), n)
        P = 0.5 * (P + P.T)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        res = np.abs(A.T @ P @ A - P
                     - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q).max()
        residuals.append(res)
        if res < tol:
            break

    return P, K, residuals


# ================================================================ #
#  예제 코드와 동일한 시스템 / 가중치
# ================================================================ #
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
n = A.shape[0]
Q = np.diag([1e2, 1e2, 1e0, 1e0])
R = 1e-2 * np.eye(2)

x0 = np.array([0.0, 0.0, 0.0, 0.0])
xref = np.array([0.5, 1.0, 0.0, 0.0])
Ts = 0.05

print("=" * 72)
print("[1] 연속시간 CARE — 반복법으로 직접 구현")
print(f"    개루프 고유값: {np.sort_complex(np.linalg.eigvals(A)).round(3)}")
print("    전부 실수부가 음수이므로 A 가 안정 -> K=0 에서 시작해도 된다.")

P_it, K_it, res_c = care_iterative(A, B, Q, R, verbose=False)
P_sp = solve_continuous_are(A, B, Q, R)
K_sp = np.linalg.solve(R, B.T @ P_sp)

print(f"\n    반복 {len(res_c)}회에 수렴")
print(f"    ||P_반복 - P_scipy||_max = {np.abs(P_it - P_sp).max():.3e}")
print(f"    ||K_반복 - K_scipy||_max = {np.abs(K_it - K_sp).max():.3e}")
print(f"\n    K =\n{np.array2string(K_it, precision=3)}")
print(f"    닫힌루프 고유값: {np.sort_complex(np.linalg.eigvals(A - B @ K_it)).round(3)}")
print("\n    잔차 추이 (원래 리카티 식에 대입한 최대 오차):")
for i, r in enumerate(res_c):
    bar = "#" * max(0, int(20 + 1.4 * np.log10(max(r, 1e-16))))
    print(f"      {i+1:>3}  {r:.2e}  {bar}")

# ---------------------------------------------------------------- #
print("-" * 72)
print("[2] 이산시간 DARE — 같은 방식으로")
Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)
Pd_it, Kd_it, res_d = dare_iterative(Ad, Bd, Q, R)
Pd_sp = solve_discrete_are(Ad, Bd, Q, R)
Kd_sp = np.linalg.solve(R + Bd.T @ Pd_sp @ Bd, Bd.T @ Pd_sp @ Ad)
print(f"    Ts = {Ts},  반복 {len(res_d)}회에 수렴")
print(f"    ||P_반복 - P_scipy||_max = {np.abs(Pd_it - Pd_sp).max():.3e}")
print(f"    ||K_반복 - K_scipy||_max = {np.abs(Kd_it - Kd_sp).max():.3e}")

# ---------------------------------------------------------------- #
print("-" * 72)
print("[3] K=0 에서 시작해도 되는 조건")
print("    반복 첫 단계는 K=0 일 때의 리아푸노프 방정식을 푼다.")
print("    그 해가 존재하려면 그 시점의 닫힌루프 A-BK = A 가 안정해야 한다.")
print("    A 가 불안정하면 K=0 으로는 시작할 수 없다. 확인해보자.")

A_un = A.copy()
A_un[2, 0] = +3.0          # 부호를 뒤집어 불안정하게 만든다
eig_un = np.linalg.eigvals(A_un)
print(f"\n    수정한 A 의 고유값: {np.sort_complex(eig_un).round(3)}")
print(f"    최대 실수부 {eig_un.real.max():+.3f} -> 불안정")

try:
    P_bad, K_bad, res_bad = care_iterative(A_un, B, Q, R, max_iter=30)
    ok = np.abs(A_un.T @ P_bad + P_bad @ A_un
                - P_bad @ B @ np.linalg.solve(R, B.T @ P_bad) + Q).max()
    pd = np.linalg.eigvalsh(P_bad).min() > 0
    print(f"    K=0 으로 시작한 결과: 잔차 {ok:.3e},  P 가 양정치인가? {pd}")
    if ok > 1e-6 or not pd:
        print("    -> 수렴하지 못했다. 예상대로다.")
except np.linalg.LinAlgError as e:
    print(f"    -> 선형계 풀이 실패: {e}")

# 안정화 이득으로 시작하면 된다
K_start = np.linalg.solve(R, B.T @ solve_continuous_are(A_un, B, np.eye(n), R))


def care_iterative_from(A, B, Q, R, K0, tol=1e-12, max_iter=200):
    nn = A.shape[0]; In = np.eye(nn); K = K0.copy(); rs = []
    for _ in range(max_iter):
        M = A - B @ K
        P = unvec(-np.linalg.solve(np.kron(In, M.T) + np.kron(M.T, In),
                                   vec(Q + K.T @ R @ K)), nn)
        P = 0.5 * (P + P.T)
        K = np.linalg.solve(R, B.T @ P)
        rs.append(np.abs(A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q).max())
        if rs[-1] < tol:
            break
    return P, K, rs


P_fix, K_fix, res_fix = care_iterative_from(A_un, B, Q, R, K_start)
P_ref = solve_continuous_are(A_un, B, Q, R)
print(f"\n    안정화 이득 K0 로 시작하면: 반복 {len(res_fix)}회, "
      f"scipy 와 차이 {np.abs(P_fix - P_ref).max():.3e}")
print("    반복법은 '이미 안정한 정책'에서 출발해야 한다는 조건이 붙는다.")
print("    scipy 의 solve_continuous_are 는 이 제약이 없다 (해밀토니안 행렬의")
print("    고유공간을 직접 분해하는 다른 방법을 쓴다).")

# ================================================================ #
#  [4] 예제 코드와 동일한 시뮬레이션 — 반복법으로 구한 K 로
# ================================================================ #
print("-" * 72)
print("[4] 반복법으로 구한 K 로 예제와 같은 시뮬레이션")

sol = solve_ivp(lambda t, x: A @ x + B @ K_it @ (xref - x),
                (0.0, 5.0), x0, t_eval=np.linspace(0.0, 5.0, 501))

t = np.arange(0.0, 5.0, Ts)
y = [x0.copy()]
for _ in t[1:]:
    y.append(Ad @ y[-1] + Bd @ Kd_it @ (xref - y[-1]))
y = np.array(y).T
print(f"    연속시간 최종 상태 {sol.y[:, -1].round(4)}")
print(f"    이산시간 최종 상태 {y[:, -1].round(4)}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    axes[0, 0].plot(sol.t, sol.y[k], lw=1.8, label=label)
    axes[0, 0].axhline(xref[k], color="gray", ls="--", lw=.8)
axes[0, 0].set(xlabel="t [s]", ylabel="state", title="(a) continuous-time LQR (iterative K)")
axes[0, 0].grid(alpha=.3); axes[0, 0].legend(fontsize=9, ncol=4)

for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    axes[0, 1].plot(t, y[k], lw=1.8, label=label)
    axes[0, 1].axhline(xref[k], color="gray", ls="--", lw=.8)
axes[0, 1].set(xlabel="t [s]", ylabel="state", title=f"(b) discrete-time LQR, Ts={Ts}")
axes[0, 1].grid(alpha=.3); axes[0, 1].legend(fontsize=9, ncol=4)

axes[1, 0].semilogy(range(1, len(res_c)+1), res_c, "o-", color="tab:blue", lw=2,
                    label=f"continuous ({len(res_c)} iters)")
axes[1, 0].semilogy(range(1, len(res_d)+1), res_d, "s-", color="tab:green", lw=2,
                    label=f"discrete ({len(res_d)} iters)")
axes[1, 0].axhline(1e-12, color="r", ls=":", lw=1.5)
axes[1, 0].text(1, 3e-12, "tolerance 1e-12", color="r", fontsize=9)
axes[1, 0].set(xlabel="iteration", ylabel="Riccati residual (max abs)",
               title="(c) convergence — note the quadratic tail")
axes[1, 0].grid(alpha=.3, which="both"); axes[1, 0].legend(fontsize=9)

axes[1, 1].semilogy(range(1, len(res_fix)+1), res_fix, "^-", color="tab:orange", lw=2,
                    label="unstable A, stabilizing K0")
if 'res_bad' in dir():
    axes[1, 1].semilogy(range(1, len(res_bad)+1), res_bad, "x--", color="tab:red", lw=2,
                        label="unstable A, K0 = 0  (fails)")
axes[1, 1].set(xlabel="iteration", ylabel="Riccati residual",
               title="(d) K0 = 0 only works if A is already stable")
axes[1, 1].grid(alpha=.3, which="both"); axes[1, 1].legend(fontsize=9)

fig.suptitle("LQR by the iterative (Kleinman) solution instead of scipy", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "lqr_iterative.png", dpi=120); plt.close(fig)

print("=" * 72)
print(f"결과 저장: {OUT / 'lqr_iterative.png'}")
