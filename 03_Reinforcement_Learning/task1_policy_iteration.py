#==========================================#
# Title:  03 RL — Task 1. Policy Iteration
#
#         환경: FrozenLake-v1 (4x4, is_slippery=True)
#               미끄러지는 얼음판. 의도한 방향으로 1/3,
#               좌우 직각 방향으로 각각 1/3 확률로 움직인다.
#               구멍(H)에 빠지면 종료, 목표(G)에 닿으면 보상 1.
#
# Author: Hyun Han
# Date:   2026-09-09
#
# Policy Iteration 은 두 단계를 번갈아 한다.
#   1) 정책 평가 (policy evaluation)  — 정책 pi 를 고정하고 V^pi 를 끝까지 구한다
#   2) 정책 개선 (policy improvement) — 그 V 로 각 상태에서 탐욕적으로 행동을 고른다
#   정책이 더 안 바뀌면 최적이다.
#
# 02 LQR 에서 리카티를 반복법으로 풀었던 구조와 정확히 같다.
#   LQR:  K 고정 -> 리아푸노프 방정식 풀어 P 구함 -> K = R^-1 B^T P 로 개선
#   RL :  pi 고정 -> 벨만 방정식 풀어 V 구함    -> pi = argmax 로 개선
# 둘 다 "평가 -> 개선"의 반복이고, 평가 단계가 선형방정식 풀이라는 것도 같다.
#==========================================#
import time

import numpy as np
import gymnasium as gym

ENV_ID   = "FrozenLake-v1"
ENV_KW   = dict(map_name="4x4", is_slippery=True)
GAMMA    = 0.99
THETA    = 1e-10          # 수렴 판정 기준
ACTIONS  = ["←", "↓", "→", "↑"]


def get_model(env_id=ENV_ID, **kw):
    """환경의 전이 모델 P[s][a] = [(확률, 다음상태, 보상, 종료), ...] 을 꺼낸다.

    DP(동적계획법)는 모델을 안다는 전제가 필요하다. FrozenLake 는 모델을 공개한다.
    모델을 모르면 Q-learning 이나 DQN 같은 model-free 방법을 써야 한다 (Task 3).
    """
    env = gym.make(env_id, **kw)
    u = env.unwrapped
    return env, u.observation_space.n, u.action_space.n, u.P


# ---------------------------------------------------------------- #
#  1) 정책 평가 — 반복법
# ---------------------------------------------------------------- #
def policy_evaluation(policy, P, nS, nA, gamma=GAMMA, theta=THETA):
    """정책 pi 를 고정하고 벨만 기대방정식을 반복해서 V^pi 를 구한다.

        V(s) <- sum_a pi(a|s) sum_s' P(s'|s,a) [ r + gamma V(s') ]
    """
    V = np.zeros(nS)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(nS):
            v_new = 0.0
            for a, pa in enumerate(policy[s]):
                if pa == 0:
                    continue
                for prob, s2, r, done in P[s][a]:
                    v_new += pa * prob * (r + gamma * V[s2] * (not done))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        sweeps += 1
        if delta < theta:
            break
    return V, sweeps


def policy_evaluation_exact(policy, P, nS, nA, gamma=GAMMA):
    """같은 것을 선형방정식으로 한 번에 푼다: (I - gamma*P_pi) V = r_pi.

    02 LQR 의 리아푸노프 방정식 풀이에 대응한다. 상태가 적을 때만 쓸 수 있다.
    """
    Ppi = np.zeros((nS, nS))
    rpi = np.zeros(nS)
    for s in range(nS):
        for a, pa in enumerate(policy[s]):
            if pa == 0:
                continue
            for prob, s2, r, done in P[s][a]:
                rpi[s] += pa * prob * r
                if not done:
                    Ppi[s, s2] += pa * prob * gamma
    return np.linalg.solve(np.eye(nS) - Ppi, rpi)


# ---------------------------------------------------------------- #
#  2) 정책 개선
# ---------------------------------------------------------------- #
def q_from_v(V, P, nS, nA, gamma=GAMMA):
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for prob, s2, r, done in P[s][a]:
                Q[s, a] += prob * (r + gamma * V[s2] * (not done))
    return Q


def policy_improvement(V, P, nS, nA, gamma=GAMMA):
    """각 상태에서 Q 가 최대인 행동을 고른다 (동점이면 균등 분배)."""
    Q = q_from_v(V, P, nS, nA, gamma)
    policy = np.zeros((nS, nA))
    for s in range(nS):
        best = np.flatnonzero(Q[s] == Q[s].max())
        policy[s, best] = 1.0 / len(best)
    return policy


# ---------------------------------------------------------------- #
def policy_iteration(P, nS, nA, gamma=GAMMA, theta=THETA, exact=False, verbose=True):
    policy = np.ones((nS, nA)) / nA            # 균등 무작위 정책에서 시작
    total_sweeps, hist = 0, []
    t0 = time.time()

    for it in range(1, 1000):
        if exact:
            V = policy_evaluation_exact(policy, P, nS, nA, gamma); sweeps = 1
        else:
            V, sweeps = policy_evaluation(policy, P, nS, nA, gamma, theta)
        total_sweeps += sweeps

        new_policy = policy_improvement(V, P, nS, nA, gamma)
        stable = np.array_equal(new_policy.argmax(1), policy.argmax(1))
        hist.append(dict(iter=it, sweeps=sweeps, V_mean=float(V.mean()),
                         V_start=float(V[0])))
        if verbose:
            print(f"    반복 {it:>2}  정책평가 sweep {sweeps:>4}회  "
                  f"V(시작상태) {V[0]:.6f}  {'정책 변화 없음' if stable else ''}")
        policy = new_policy
        if stable:
            break

    elapsed = time.time() - t0
    stats = dict(outer_iters=it, total_sweeps=total_sweeps,
                 backups=total_sweeps * nS * nA, seconds=round(elapsed, 4), hist=hist)
    return V, policy, stats


# ---------------------------------------------------------------- #
def render_policy(policy, desc, nS):
    """정책을 격자 그림으로."""
    n = int(np.sqrt(nS))
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            ch = desc[r][c].decode()
            row.append(ch if ch in "HG" else ACTIONS[policy[r * n + c].argmax()])
        grid.append(" ".join(f"{x:>2}" for x in row))
    return "\n".join("      " + g for g in grid)


def evaluate_policy(env_id, env_kw, policy, episodes=2000, seed=0):
    """실제로 환경을 돌려 성공률을 잰다 (이론값 V(0) 과 대조용)."""
    env = gym.make(env_id, **env_kw)
    rng = np.random.default_rng(seed)
    wins = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            a = int(policy[s].argmax())
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            wins += r
    env.close()
    return wins / episodes


def main():
    print("=" * 74)
    print("Task 1. Policy Iteration —", ENV_ID, ENV_KW)
    env, nS, nA, P = get_model(**ENV_KW)
    print(f"상태 {nS}개, 행동 {nA}개, gamma={GAMMA}")
    print(f"미끄러짐: 의도한 방향으로 1/3, 좌우 직각으로 각 1/3\n")

    V, policy, stats = policy_iteration(P, nS, nA)

    print(f"\n  수렴: 바깥 반복 {stats['outer_iters']}회, "
          f"정책평가 sweep 총 {stats['total_sweeps']}회")
    print(f"  벨만 백업 {stats['backups']:,}회, {stats['seconds']:.3f}초")
    print(f"\n  V(시작상태) = {V[0]:.6f}   <- 최적정책의 이론적 성공 가치")
    print("\n  최적 정책:")
    print(render_policy(policy, env.unwrapped.desc, nS))

    rate = evaluate_policy(ENV_ID, ENV_KW, policy)
    print(f"\n  실제 2000 에피소드 성공률: {rate*100:.1f}%")

    # 정확 해법과 대조
    V_ex, _, st_ex = policy_iteration(P, nS, nA, exact=True, verbose=False)
    print(f"\n  [참고] 정책평가를 선형방정식으로 직접 풀면: "
          f"바깥 반복 {st_ex['outer_iters']}회, {st_ex['seconds']:.4f}초")
    print(f"         V 차이 {np.abs(V - V_ex).max():.2e} — 같은 답")
    print("         02 LQR 에서 리아푸노프 방정식을 푼 것과 같은 자리다.")
    return V, policy, stats


if __name__ == "__main__":
    main()
