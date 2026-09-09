#==========================================#
# Title:  03 RL — Task 3. stable-baselines3 로 학습
#
#         환경: CartPole-v1
#         모델: DQN (비교용으로 PPO 도 같은 예산으로 함께 학습)
#
# Author: Hyun Han
# Date:   2026-09-09
#
# Task 1·2 와 근본적으로 다른 상황이다.
#
#   FrozenLake : 상태 16개, 전이확률 P 를 환경이 공개한다 -> 표를 채우면 끝 (DP)
#   CartPole   : 상태가 연속 4차원 (위치, 속도, 각도, 각속도)
#                -> 표를 만들 수 없다. 전이 모델도 모른다.
#
# 그래서 두 가지가 바뀐다.
#   1) V 를 표가 아니라 신경망으로 근사한다 (function approximation)
#   2) 모델을 모르니 직접 굴려본 경험으로 배운다 (model-free)
#
# DQN 은 Q-learning + 신경망이고, 그것만으로는 발산해서 두 장치를 덧붙였다.
#   - Replay buffer : 과거 경험을 섞어 뽑아 시간적 상관을 깬다
#   - Target network : 목표값을 만드는 망을 잠시 고정해 과녁이 흔들리지 않게 한다
#
# 사용법:  python task3_dqn.py
#          COMPARE=0 python task3_dqn.py     <- PPO 비교 생략 (DQN 만)
#==========================================#
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID     = "CartPole-v1"
TIMESTEPS  = 60_000
SEED       = 42
COMPARE    = os.environ.get("COMPARE", "1") == "1"

HERE = Path(__file__).resolve().parent
OUT  = HERE / "results"
OUT.mkdir(exist_ok=True)


class RewardLogger(BaseCallback):
    """에피소드가 끝날 때마다 (timestep, 총보상, 탐험률) 을 기록한다."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                eps = getattr(self.model, "exploration_rate", float("nan"))
                self.rows.append((self.num_timesteps, float(ep["r"]), float(eps)))
        return True


def make_env(seed=SEED):
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed)
    return env


def train(algo_name):
    """DQN 또는 PPO 를 같은 예산으로 학습한다."""
    env = make_env()
    if algo_name == "DQN":
        # stable-baselines3 RL Zoo 의 CartPole 튜닝값
        model = DQN("MlpPolicy", env, seed=SEED, verbose=0,
                    learning_rate=2.3e-3, batch_size=64, buffer_size=100_000,
                    learning_starts=1_000, gamma=0.99,
                    target_update_interval=10,          # 목표망 갱신 주기
                    train_freq=256, gradient_steps=128,
                    exploration_fraction=0.16,          # 앞 16% 구간에서 eps 를 낮춘다
                    exploration_final_eps=0.04,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", env, seed=SEED, verbose=0,
                    n_steps=1024, batch_size=64, gae_lambda=0.98,
                    gamma=0.999, n_epochs=4, ent_coef=0.01)

    # 학습 중 주기적으로 '탐험 없이' 평가하고, 가장 좋았던 시점의 가중치를 남긴다.
    # DQN 은 학습 후반에 성능이 무너지는 일이 흔해서 이 장치가 사실상 필수다.
    # (001 MLP 에서 best validation 가중치를 저장한 것과 같은 발상)
    log_dir = OUT / f"eval_{algo_name.lower()}"
    eval_cb = EvalCallback(make_env(seed=SEED + 50), best_model_save_path=str(log_dir),
                           log_path=str(log_dir), eval_freq=2_500,
                           n_eval_episodes=5, deterministic=True, verbose=0)
    cb = RewardLogger()
    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=[cb, eval_cb], progress_bar=False)
    secs = time.time() - t0

    eval_env = make_env(seed=SEED + 100)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=30,
                                    deterministic=True)

    # 최고 시점 가중치를 불러와 다시 평가한다 (제출용 성능)
    cls = DQN if algo_name == "DQN" else PPO
    best = cls.load(str(log_dir / "best_model"))
    b_mean, b_std = evaluate_policy(best, eval_env, n_eval_episodes=30,
                                    deterministic=True)
    ev = np.load(log_dir / "evaluations.npz")
    env.close(); eval_env.close()

    rows = np.array(cb.rows) if cb.rows else np.zeros((0, 3))
    return best, rows, dict(algo=algo_name, seconds=round(secs, 1),
                            episodes=len(rows),
                            final_eval_mean=round(float(mean_r), 2),
                            final_eval_std=round(float(std_r), 2),
                            best_eval_mean=round(float(b_mean), 2),
                            best_eval_std=round(float(b_std), 2)), \
           (ev["timesteps"], ev["results"].mean(1))


def moving_average(x, w=20):
    if len(x) < w:
        return np.array(x, float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    print("=" * 74)
    print(f"Task 3. stable-baselines3 —  환경 {ENV_ID},  예산 {TIMESTEPS:,} timesteps")
    print("CartPole-v1: 관측 4차원 연속(위치·속도·각도·각속도), 행동 2개(좌/우)")
    print("             매 스텝 +1 보상, 최대 500. 각도 12도 초과 또는 위치 이탈 시 종료.")
    print("             상태가 연속이라 Task 1·2 의 표 방식은 쓸 수 없다.\n")

    results, curves, evals = {}, {}, {}
    algos = ["DQN"] + (["PPO"] if COMPARE else [])
    for a in algos:
        print(f"  [{a}] 학습 중...", end=" ", flush=True)
        model, rows, st, ev = train(a)
        results[a], curves[a], evals[a] = st, rows, ev
        print(f"{st['seconds']:.0f}초, 에피소드 {st['episodes']}개")
        print(f"        마지막 시점 {st['final_eval_mean']:6.1f} ± {st['final_eval_std']:.1f}"
              f"   |   최고 시점 {st['best_eval_mean']:6.1f} ± {st['best_eval_std']:.1f}")
        model.save(OUT / f"{a.lower()}_cartpole")

    # ------------------------------------------------------------ #
    #  DQN 학습 곡선을 구간별로 뜯어본다 (설명 2 의 근거)
    # ------------------------------------------------------------ #
    d = curves["DQN"]
    print("\n" + "-" * 74)
    print("[DQN] 학습 구간별 평균 보상과 탐험률")
    print(f"{'timestep 구간':>18}{'에피소드':>10}{'평균 보상':>12}{'탐험률 eps':>13}")
    phases = []
    edges = np.linspace(0, TIMESTEPS, 7).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d[:, 0] >= lo) & (d[:, 0] < hi)
        if m.sum() == 0:
            continue
        phases.append(dict(lo=int(lo), hi=int(hi), n=int(m.sum()),
                           mean_r=round(float(d[m, 1].mean()), 1),
                           eps=round(float(d[m, 2].mean()), 3)))
        print(f"{f'{lo:,} ~ {hi:,}':>18}{m.sum():>10}"
              f"{d[m,1].mean():>12.1f}{d[m,2].mean():>13.3f}")
    results["DQN"]["phases"] = phases
    results["DQN"]["best_episode"] = float(d[:, 1].max())
    results["DQN"]["last50_mean"] = round(float(d[-50:, 1].mean()), 1)

    # ------------------------------------------------------------ #
    #  reward graph (제출물)
    # ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(d[:, 0], d[:, 1], color="tab:blue", alpha=.22, lw=.8, label="episode reward")
    ma = moving_average(d[:, 1], 20)
    ax.plot(d[len(d) - len(ma):, 0], ma, color="tab:blue", lw=2.4,
            label="moving average (20 ep)")
    ax.axhline(500, color="k", ls="--", lw=1.2, label="max score (500)")
    ax.axhline(475, color="tab:green", ls=":", lw=1.5, label="'solved' (475)")
    ax2 = ax.twinx()
    ax2.plot(d[:, 0], d[:, 2], color="crimson", lw=1.4, ls="-.")
    ax2.set_ylabel("exploration rate  $\\epsilon$", color="crimson")
    ax2.tick_params(axis="y", colors="crimson"); ax2.set_ylim(0, 1.05)
    ax.set(xlabel="timesteps", ylabel="episode reward", ylim=(0, 540),
           title="(a) DQN on CartPole-v1 — reward curve")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for a, c in zip(algos, ("tab:blue", "tab:orange")):
        t, r = evals[a]
        ax.plot(t, r, "o-", color=c, lw=2.2, ms=4,
                label=f"{a}  best {results[a]['best_eval_mean']:.0f} / "
                      f"last {results[a]['final_eval_mean']:.0f}")
        ax.scatter([t[r.argmax()]], [r.max()], color=c, s=120, marker="*",
                   zorder=5, edgecolor="k", linewidth=.6)
    ax.axhline(500, color="k", ls="--", lw=1.2)
    ax.set(xlabel="timesteps", ylabel="evaluation reward (greedy, 5 ep)", ylim=(0, 540),
           title="(b) greedy evaluation during training\n"
                 "star = best checkpoint (saved)")
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Task 3 — stable-baselines3 on CartPole-v1", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "task3_reward.png", dpi=120); plt.close(fig)

    for a in algos:
        np.savetxt(OUT / f"rewards_{a.lower()}.csv", curves[a],
                   delimiter=",", header="timestep,reward,epsilon", comments="")
    with open(OUT / "task3_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(env=ENV_ID, timesteps=TIMESTEPS, seed=SEED, results=results),
                  f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 74)
    for a in algos:
        r = results[a]
        print(f"{a:<5} 최고 시점 {r['best_eval_mean']:6.1f} ± {r['best_eval_std']:5.1f}"
              f"   |  마지막 시점 {r['final_eval_mean']:6.1f}"
              f"   |  학습 {r['seconds']:.0f}초")
    print(f"\nreward graph: {OUT/'task3_reward.png'}   <- 제출물")


if __name__ == "__main__":
    main()
