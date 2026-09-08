#==========================================#
# Title:  01-003 RNN & Regression — 학습 및 평가
#
#         LSTM 세 종류를 같은 데이터로 학습해 비교한다.
#           m2o          : 과거 60일 -> 다음 1일
#           m2m          : 과거 60일 -> 앞으로 5일   (many-to-many)
#           m2m_aligned  : 과거 60일 -> 각 시점의 다음날 60개 (many-to-many)
#
#         그리고 셋 다 '아무것도 안 하는 기준선'과 비교한다.
#         주가 예측에서 이 비교를 빼면 결과를 해석할 수 없다.
#
# Author: Hyun Han
# Date:   2026-09-09
#
# 사용법:  python train.py
#==========================================#
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import prepare
from rnn_network import build

SEED     = 42
SEQ_LEN  = 60
HORIZON  = 5
EPOCHS   = 120
BATCH    = 64
LR       = 1e-3
PATIENCE = 20              # val 이 이만큼 안 좋아지면 조기 종료

HERE = Path(__file__).resolve().parent
OUT  = HERE / "results"
OUT.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- #
def run_epoch(model, loader, lossfn, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = lossfn(pred, yb)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # RNN 은 폭발 방지
                opt.step()
            tot += loss.item() * len(xb); n += len(xb)
    return tot / n


def fit(kind, d, verbose=False):
    torch.manual_seed(SEED)
    n_feat = d[kind]["train"][0].shape[2]
    model = build(kind, n_feat, HORIZON).to(device)

    def mk(split, shuffle):
        X, y, _ = d[kind][split]
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                          batch_size=BATCH, shuffle=shuffle)

    tr_loader, va_loader = mk("train", True), mk("val", False)
    lossfn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    hist = {"train": [], "val": []}
    best, best_state, bad, best_ep = float("inf"), None, 0, 0
    for ep in range(EPOCHS):
        tr = run_epoch(model, tr_loader, lossfn, opt)
        va = run_epoch(model, va_loader, lossfn)
        hist["train"].append(tr); hist["val"].append(va)
        if va < best - 1e-6:
            best, best_ep, bad = va, ep + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if verbose and (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:>3}  train {tr:.4f}  val {va:.4f}")
    model.load_state_dict(best_state)
    return model, hist, best_ep


# ---------------------------------------------------------------- #
#  평가 — 스케일을 되돌려 실제 수익률 단위로 본다
# ---------------------------------------------------------------- #
@torch.no_grad()
def predict(model, d, kind, split):
    X, y, idx = d[kind][split]
    model.eval()
    p = model(torch.tensor(X).to(device)).cpu().numpy()
    return p * d["y_sd"], y * d["y_sd"], idx


def significance(acc_pct, n, p0_pct):
    """방향 정확도가 기준선 p0 보다 나은 게 우연일 확률 (양측 이항검정의 정규근사).

    n 회 동전을 던져 p0 의 확률로 맞히는 게 기준선일 때,
    관측된 acc 가 얼마나 벗어나 있는지를 표준편차 단위(z)로 잰다.
    |z| < 1.96 이면 p > 0.05 — 우연으로 설명 가능하다는 뜻이다.
    """
    p, p0 = acc_pct / 100, p0_pct / 100
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval


def metrics(pred, true):
    """RMSE(수익률), 방향 정확도. 둘 다 1차원으로 펴서 계산."""
    p, t = np.asarray(pred).ravel(), np.asarray(true).ravel()
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mask = t != 0
    da = float(np.mean(np.sign(p[mask]) == np.sign(t[mask])) * 100)
    return rmse, da


# ---------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 74)
    print(f"device: {device}")
    d = prepare(seq_len=SEQ_LEN, horizon=HORIZON)
    summary = {"seq_len": SEQ_LEN, "horizon": HORIZON,
               "features": d["feature_names"], "device": str(device)}

    # ------------------------------------------------------------ #
    #  기준선 — "내일은 오늘과 같다" (수익률 0 으로 예측)
    #  주가 수익률은 거의 백색잡음이라 이 기준선이 의외로 강하다.
    #  RMSE 로는 이걸 이기기가 매우 어렵다.
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    _, y_te, _ = d["m2o"]["test"]
    y_te = (y_te * d["y_sd"]).ravel()
    n_test = len(y_te)
    base_rmse = float(np.sqrt(np.mean(y_te ** 2)))
    up_ratio = float(np.mean(y_te > 0) * 100)

    print(f"[기준선 1] '내일 = 오늘'  (수익률 0 예측)")
    print(f"           RMSE {base_rmse:.5f}   방향 정확도 50.00% (동전 던지기)")
    print(f"[기준선 2] '무조건 오른다' (항상 + 로 예측)          <-- 진짜 기준선")
    print(f"           방향 정확도 {up_ratio:.2f}%   "
          f"(test {n_test}일 중 오른 날의 비율)")
    print( "           주가는 장기적으로 오르는 쪽이 많다. 50% 를 이겼다고")
    print( "           좋아하면 안 되고, 이 값을 이겨야 무언가를 배운 것이다.")
    summary["baseline"] = dict(rmse=round(base_rmse, 6), dir_acc_coin=50.0,
                               dir_acc_always_up=round(up_ratio, 2),
                               n_test=n_test)

    # ------------------------------------------------------------ #
    #  세 모델 학습
    # ------------------------------------------------------------ #
    names = {"m2o": "many-to-one (60일 -> 1일)",
             "m2m": f"many-to-many (60일 -> {HORIZON}일)",
             "m2m_aligned": "many-to-many aligned (60일 -> 60일)"}
    models, hists, results = {}, {}, {}

    print("\n" + "-" * 74)
    for kind in ("m2o", "m2m", "m2m_aligned"):
        t0 = time.time()
        model, hist, best_ep = fit(kind, d)
        p_te, t_te, _ = predict(model, d, kind, "test")

        if kind == "m2m_aligned":
            p_ev, t_ev = p_te[:, -1], t_te[:, -1]      # 마지막 시점만 평가
        elif kind == "m2m":
            p_ev, t_ev = p_te[:, 0], t_te[:, 0]        # 1일 후만 (m2o 와 공정 비교)
        else:
            p_ev, t_ev = p_te[:, 0], t_te[:, 0]

        rmse, da = metrics(p_ev, t_ev)
        n_ev = len(np.asarray(p_ev).ravel())
        z_coin, p_coin = significance(da, n_ev, 50.0)
        z_up,   p_up   = significance(da, n_ev, up_ratio)
        n_par = sum(p.numel() for p in model.parameters())
        results[kind] = dict(name=names[kind], rmse=round(rmse, 6),
                             dir_acc=round(da, 2), params=n_par,
                             best_epoch=best_ep, epochs_run=len(hist["train"]),
                             n_eval=n_ev,
                             p_vs_coin=round(p_coin, 4), p_vs_always_up=round(p_up, 4),
                             beats_always_up=bool(da > up_ratio and p_up < 0.05),
                             train_s=round(time.time() - t0, 1))
        models[kind], hists[kind] = model, hist
        print(f"[{kind:<11}] {names[kind]:<34} "
              f"RMSE {rmse:.5f} ({rmse/base_rmse-1:+.1%})  "
              f"방향 {da:5.2f}%   {time.time()-t0:.0f}초")
        print(f"{'':<14} 동전(50%) 대비 p={p_coin:.3f}"
              f"{'  유의함' if p_coin < 0.05 else '  우연일 수 있음'}"
              f"   |   항상상승({up_ratio:.1f}%) 대비 p={p_up:.3f}"
              f"{'  유의함' if p_up < 0.05 and da > up_ratio else '  못 이김'}")
    summary["models"] = results

    # ------------------------------------------------------------ #
    #  many-to-many 는 며칠 앞까지 쓸 만한가
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    print("[예측 지평] many-to-many 모델이 h일 뒤를 얼마나 맞히나")
    p_te, t_te, _ = predict(models["m2m"], d, "m2m", "test")
    horizon_tbl = {}
    print(f"{'h일 뒤':>8}{'RMSE':>12}{'방향 정확도':>14}")
    for h in range(HORIZON):
        r, a = metrics(p_te[:, h], t_te[:, h])
        horizon_tbl[h + 1] = dict(rmse=round(r, 6), dir_acc=round(a, 2))
        print(f"{h+1:>7}일{r:>12.5f}{a:>13.2f}%")
    summary["horizon"] = horizon_tbl

    # ------------------------------------------------------------ #
    #  그림
    # ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # (a) 학습곡선
    for kind, c in zip(("m2o", "m2m", "m2m_aligned"),
                       ("tab:blue", "tab:orange", "tab:green")):
        h = hists[kind]
        axes[0].plot(h["train"], color=c, ls="--", lw=1.2, alpha=.6)
        axes[0].plot(h["val"], color=c, lw=2, label=kind)
    axes[0].set(xlabel="epoch", ylabel="MSE (scaled)", yscale="log",
                title="(a) learning curves\n(dashed = train, solid = val)")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=9)

    # (b) test 구간 가격 재구성
    p_te, t_te, idx = predict(models["m2o"], d, "m2o", "test")
    close = d["close"]
    prev = close[idx]                              # 예측 기준일의 종가
    price_true = close[idx + 1]                    # 실제 다음날 종가
    price_pred = prev * np.exp(p_te[:, 0])         # 모델 예측
    axes[1].plot(price_true, color="k", lw=1.8, label="actual")
    axes[1].plot(price_pred, color="tab:blue", lw=1.3, alpha=.85, label="LSTM (m2o)")
    axes[1].plot(prev, color="tab:red", lw=1.0, ls=":", label="naive (= today)")
    axes[1].set(xlabel="test day", ylabel="close price",
                title="(b) test period — all three look identical\n"
                      "that is the point, not a success")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=9)

    # (c) 지평별 성능 + 모델 비교
    hs = list(horizon_tbl.keys())
    axes[2].bar([h - 0.2 for h in hs], [horizon_tbl[h]["dir_acc"] for h in hs],
                width=.4, color="tab:orange", label="m2m directional acc.")
    axes[2].axhline(50, color="k", ls="--", lw=1.5)
    axes[2].text(hs[0] - .45, 50.4, "coin flip (50%)", fontsize=8)
    axes[2].axhline(up_ratio, color="crimson", ls="-.", lw=1.8)
    axes[2].text(hs[0] - .45, up_ratio + 0.4,
                 f"always-up ({up_ratio:.1f}%)  <- real baseline",
                 fontsize=8, color="crimson")
    for kind, c, off in [("m2o", "tab:blue", 0.2), ("m2m_aligned", "tab:green", 0.6)]:
        axes[2].bar([hs[0] + off], [results[kind]["dir_acc"]], width=.4,
                    color=c, label=f"{kind} (1-day)")
    axes[2].set(xlabel="prediction horizon [days]", ylabel="directional accuracy [%]",
                ylim=(35, 70), xticks=hs,
                title="(c) can it call the direction?")
    axes[2].grid(alpha=.3, axis="y"); axes[2].legend(fontsize=8)

    fig.suptitle("LSTM stock return prediction — many-to-one vs many-to-many", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "rnn_stock.png", dpi=120); plt.close(fig)

    for kind, m in models.items():
        torch.save(m.state_dict(), OUT / f"lstm_{kind}.pt")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------ #
    best = max(results, key=lambda k: results[k]["dir_acc"])
    print("\n" + "=" * 74)
    print(f"가장 좋은 모델: {names[best]}")
    r = results[best]
    print(f"  방향 정확도 {r['dir_acc']:.2f}%  (노션 리더보드에 이 값)")
    print(f"  RMSE {r['rmse']:.5f}  vs 기준선 {base_rmse:.5f}")
    print(f"  '항상 상승' 기준선 {up_ratio:.2f}% 를 "
          f"{'이겼다' if r['beats_always_up'] else '이기지 못했다'} "
          f"(p={r['p_vs_always_up']:.3f})")
    print(f"결과 저장: {OUT}")


if __name__ == "__main__":
    main()
