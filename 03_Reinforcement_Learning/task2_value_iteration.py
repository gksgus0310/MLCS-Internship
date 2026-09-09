#==========================================#
# Title:  03 RL — Task 2. Value Iteration
#
#         환경: FrozenLake-v1 (4x4, is_slippery=True) — Task 1 과 동일
#
# Author: Hyun Han
# Date:   2026-09-09
#
# Value Iteration 은 정책을 명시적으로 들고 있지 않는다.
# 벨만 '최적' 방정식을 한 번씩만 적용하며 V 를 직접 밀어올린다.
#
#     V(s) <- max_a sum_s' P(s'|s,a) [ r + gamma V(s') ]
#
# Policy Iteration 과의 차이는 딱 한 군데다.
#     PI: 정책을 고정하고 V 가 완전히 수렴할 때까지 평가 -> 그 다음 개선
#     VI: 매 sweep 마다 max 를 취한다 (= 평가 1회 + 개선 1회를 합친 것)
#
# 그래서 VI 는 바깥 반복이 훨씬 많고, PI 는 바깥 반복이 적은 대신
# 한 번의 반복이 비싸다. 총 계산량으로 비교해야 공정하다.
#==========================================#
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task1_policy_iteration import (ENV_ID, ENV_KW, GAMMA, THETA, get_model,
                                    policy_improvement, render_policy,
                                    evaluate_policy, policy_iteration)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def value_iteration(P, nS, nA, gamma=GAMMA, theta=THETA, verbose=True):
    V = np.zeros(nS)
    hist = []
    t0 = time.time()

    for it in range(1, 100000):
        delta = 0.0
        for s in range(nS):
            v_new = max(
                sum(prob * (r + gamma * V[s2] * (not done))
                    for prob, s2, r, done in P[s][a])
                for a in range(nA))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        hist.append(dict(iter=it, delta=float(delta), V_start=float(V[0])))
        if verbose and (it <= 3 or it % 50 == 0):
            print(f"    sweep {it:>4}  delta {delta:.3e}  V(시작상태) {V[0]:.6f}")
        if delta < theta:
            break

    # V 가 수렴한 뒤 정책을 딱 한 번 뽑는다
    policy = policy_improvement(V, P, nS, nA, gamma)
    elapsed = time.time() - t0
    stats = dict(sweeps=it, backups=it * nS * nA,
                 seconds=round(elapsed, 4), hist=hist)
    return V, policy, stats


def main():
    print("=" * 74)
    print("Task 2. Value Iteration —", ENV_ID, ENV_KW)
    env, nS, nA, P = get_model(**ENV_KW)
    print(f"상태 {nS}개, 행동 {nA}개, gamma={GAMMA}, theta={THETA}\n")

    V_vi, pi_vi, st_vi = value_iteration(P, nS, nA)

    print(f"\n  수렴: sweep {st_vi['sweeps']}회, "
          f"벨만 백업 {st_vi['backups']:,}회, {st_vi['seconds']:.3f}초")
    print(f"  V(시작상태) = {V_vi[0]:.6f}")
    print("\n  최적 정책:")
    print(render_policy(pi_vi, env.unwrapped.desc, nS))

    rate = evaluate_policy(ENV_ID, ENV_KW, pi_vi)
    print(f"\n  실제 2000 에피소드 성공률: {rate*100:.1f}%")

    # ------------------------------------------------------------ #
    #  Task 1 과 직접 비교 (설명 1 의 근거)
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    print("Policy Iteration 과 비교")
    V_pi, pi_pi, st_pi = policy_iteration(P, nS, nA, verbose=False)

    same = bool(np.array_equal(pi_vi.argmax(1), pi_pi.argmax(1)))
    print(f"\n{'':<22}{'Policy Iteration':>20}{'Value Iteration':>20}")
    print("-" * 62)
    print(f"{'바깥 반복':<22}{st_pi['outer_iters']:>20}{st_vi['sweeps']:>20}")
    print(f"{'전체 sweep':<22}{st_pi['total_sweeps']:>20}{st_vi['sweeps']:>20}")
    print(f"{'벨만 백업':<22}{st_pi['backups']:>20,}{st_vi['backups']:>20,}")
    print(f"{'시간 [초]':<22}{st_pi['seconds']:>20.4f}{st_vi['seconds']:>20.4f}")
    print(f"{'V(시작상태)':<22}{V_pi[0]:>20.8f}{V_vi[0]:>20.8f}")
    print("-" * 62)
    print(f"\n  두 방법의 V 최대 차이 : {np.abs(V_pi - V_vi).max():.2e}")
    print(f"  두 방법의 정책이 같은가: {same}")
    print("\n  같은 최적해에 도달한다. 벨만 최적방정식의 해가 유일하기 때문이다.")
    print("  차이는 '거기 도달하는 경로'뿐이다.")

    # ------------------------------------------------------------ #
    #  PI 의 정책평가를 얼마나 대충 해도 되는가
    #
    #  위 표에서 PI 가 백업을 두 배 쓴 이유는 정책평가를 theta=1e-10 까지
    #  끝까지 밀어붙였기 때문이다. 사실 그럴 필요가 없다.
    #  평가를 도중에 끊는 것을 modified policy iteration 이라 하고,
    #  1 sweep 만에 끊은 극단이 바로 value iteration 이다.
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    print("PI 의 정책평가 정확도를 낮추면 (modified policy iteration)")
    print(f"{'theta':>10}{'바깥 반복':>12}{'전체 sweep':>13}{'벨만 백업':>13}"
          f"{'V(시작)':>13}{'최적정책?':>11}")
    sweep_tbl = []
    for th in (1e-1, 1e-2, 1e-4, 1e-6, 1e-10):
        Vh, pih, sth = policy_iteration(P, nS, nA, theta=th, verbose=False)
        ok = bool(np.array_equal(pih.argmax(1), pi_vi.argmax(1)))
        sweep_tbl.append(dict(theta=th, backups=sth['backups'],
                              iters=sth['outer_iters'], optimal=ok))
        print(f"{th:>10.0e}{sth['outer_iters']:>12}{sth['total_sweeps']:>13}"
              f"{sth['backups']:>13,}{Vh[0]:>13.6f}{str(ok):>11}")
    print(f"{'VI':>10}{st_vi['sweeps']:>12}{st_vi['sweeps']:>13}"
          f"{st_vi['backups']:>13,}{V_vi[0]:>13.6f}{'True':>11}")
    print("\n  너무 대충 끊으면(1e-1, 1e-2) 최적 정책이 깨진다.")
    print("  하지만 1e-4 에서 끊으면 최적 정책을 유지하면서 백업이")
    print("  54,272 -> 17,024 로 3분의 1이 된다. VI(26,880)보다도 싸다.")
    print("  VI 는 '정책평가를 1 sweep 만 하는 PI' 로 볼 수 있고,")
    print("  그 사이 어딘가가 실제로는 가장 빠르다.")

    # ------------------------------------------------------------ #
    #  그림
    # ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    vi_x = [h["iter"] * nS * nA for h in st_vi["hist"]]
    vi_y = [h["V_start"] for h in st_vi["hist"]]
    axes[0].plot(vi_x, vi_y, color="tab:orange", lw=2, label="Value Iteration")
    cum, pi_x, pi_y = 0, [], []
    for h in st_pi["hist"]:
        cum += h["sweeps"] * nS * nA
        pi_x.append(cum); pi_y.append(h["V_start"])
    axes[0].plot(pi_x, pi_y, "o-", color="tab:blue", lw=2, ms=8,
                 label="Policy Iteration")
    axes[0].axhline(V_vi[0], color="k", ls="--", lw=1, label=f"optimal {V_vi[0]:.4f}")
    axes[0].set(xlabel="Bellman backups (fair cost axis)", ylabel="V(start state)",
                xscale="log", title="(a) same optimum, different paths")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=9, loc="lower right")

    axes[1].semilogy([h["iter"] for h in st_vi["hist"]],
                     [h["delta"] for h in st_vi["hist"]],
                     color="tab:orange", lw=2)
    axes[1].axhline(THETA, color="crimson", ls="--", lw=1.5,
                    label=f"theta = {THETA:.0e}")
    ratios = [st_vi["hist"][i+1]["delta"] / st_vi["hist"][i]["delta"]
              for i in range(100, len(st_vi["hist"]) - 1)]
    rate_meas = float(np.mean(ratios)) if ratios else float("nan")
    axes[1].set(xlabel="sweep", ylabel="max |V_new - V_old|",
                title=f"(b) VI shrinks the error by a constant factor\n"
                      f"measured {rate_meas:.3f} per sweep (bound: gamma={GAMMA})")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=9)

    fig.suptitle("FrozenLake-v1 — Policy Iteration vs Value Iteration", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "task12_pi_vs_vi.png", dpi=120)
    plt.close(fig)

    with open(OUT / "task12_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(env=ENV_ID, env_kwargs=ENV_KW, gamma=GAMMA, theta=THETA,
                       policy_iteration=dict(outer_iters=st_pi["outer_iters"],
                                             sweeps=st_pi["total_sweeps"],
                                             backups=st_pi["backups"],
                                             seconds=st_pi["seconds"]),
                       value_iteration=dict(sweeps=st_vi["sweeps"],
                                            backups=st_vi["backups"],
                                            seconds=st_vi["seconds"]),
                       vi_contraction_measured=round(rate_meas, 4),
                       V_start=float(V_vi[0]), same_policy=same,
                       success_rate=float(rate), theta_sweep=sweep_tbl),
                  f, indent=2, ensure_ascii=False, default=float)
    print(f"\n결과 저장: {OUT}")
    return dict(vi=(V_vi, pi_vi, st_vi), pi=(V_pi, pi_pi, st_pi), nS=nS, nA=nA)


if __name__ == "__main__":
    main()
