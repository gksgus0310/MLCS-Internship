#==========================================#
# Title:  01-002 CNN & Classification — 학습 및 실험
#
#   Step 1. 사전학습 ResNet18 로 CIFAR-10 feature 를 '한 번만' 뽑아 캐시
#   Step 2. 캐시된 feature 로 분류 헤드만 학습
#   Step 3. 사전학습 있음 vs 없음 비교 — 전이학습의 값어치
#   Step 4. 학습 데이터 양을 줄여가며 비교 (캐시 덕에 거의 공짜)
#
# Author: Hyun Han
# Date:   2026-09-09
#
# 사용법:  python train.py
#          (워커 문제가 나면  $env:NUM_WORKERS=0 ; python train.py)
#
# Windows 주의: DataLoader(num_workers>0) 는 spawn 으로 워커를 띄우고,
# 워커는 이 파일을 다시 import 한다. 그래서 실행 코드가 전부
# main() 안에 있고 아래 __main__ 가드로 막혀 있어야 한다.
#==========================================#
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_network import (SEED, CLASSES, SHARED, OUT, device,
                         get_backbone, ClassifierHead, extract_features)

CACHE_PRE  = SHARED / "cifar10_resnet18_fp16.npz"
CACHE_RAND = SHARED / "cifar10_resnet18_random_fp16.npz"
N_VAL = 5000


# ---------------------------------------------------------------- #
#  헤드 학습 — 입력이 이미 512차원 feature 라 GPU 에 통째로 올라간다
# ---------------------------------------------------------------- #
def train_head(Xtr, ytr, Xva, yva, Xte, yte, epochs=30, lr=1e-3, verbose=False):
    torch.manual_seed(SEED)
    head = ClassifierHead(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=512, shuffle=True)
    Xva_d, yva_d = Xva.to(device), yva.to(device)
    Xte_d, yte_d = Xte.to(device), yte.to(device)

    hist, best_va, best_state = {"train": [], "val": []}, -1.0, None
    for ep in range(epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossfn(head(xb), yb).backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            tr_acc = (head(Xtr[:10000].to(device)).argmax(1)
                      == ytr[:10000].to(device)).float().mean().item() * 100
            va_acc = (head(Xva_d).argmax(1) == yva_d).float().mean().item() * 100
        hist["train"].append(tr_acc)
        hist["val"].append(va_acc)

        if va_acc > best_va:                       # validation 기준 모델 선택
            best_va = va_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"           epoch {ep+1:>2}/{epochs}  "
                  f"train {tr_acc:5.2f}%  val {va_acc:5.2f}%")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte_d).argmax(1)
        te_acc = (pred == yte_d).float().mean().item() * 100
    return te_acc, best_va, hist, pred.cpu().numpy(), head


# ---------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    summary = {}

    print("=" * 74)
    print(f"device: {device}"
          f"{'  (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    # ============================================================ #
    #  Step 1. feature 캐시
    # ============================================================ #
    print("\n[Step 1] ImageNet 사전학습 ResNet18 로 feature 추출")
    cache = extract_features(get_backbone(pretrained=True), CACHE_PRE)

    Xtr = torch.tensor(cache["X_train"], dtype=torch.float32)
    ytr = torch.tensor(cache["y_train"], dtype=torch.long)
    Xte = torch.tensor(cache["X_test"],  dtype=torch.float32)
    yte = torch.tensor(cache["y_test"],  dtype=torch.long)
    print(f"        train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}")
    summary["cache"] = dict(file=CACHE_PRE.name,
                            mb=round(CACHE_PRE.stat().st_size / 1024**2, 1),
                            feature_dim=int(Xtr.shape[1]))

    # train 을 train/val 로 나눈다 (001 MLP 와 같은 규칙)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(Xtr), generator=g)
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    Xva, yva = Xtr[val_idx], ytr[val_idx]
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    print(f"        train {len(Xtr)} / val {len(Xva)} / test {len(Xte)}")

    # ============================================================ #
    #  Step 2 & 3. 사전학습 있음 vs 없음
    # ============================================================ #
    print("\n" + "-" * 74)
    print("[Step 2·3] 사전학습 가중치가 실제로 값어치가 있나?")
    print("           같은 백본 구조로 (a) ImageNet 사전학습  (b) 랜덤 초기화")
    print("           둘 다 backbone 은 freeze 하고 헤드만 학습한다.\n")

    rand = extract_features(get_backbone(pretrained=False), CACHE_RAND,
                            want_labels=False, tag=" [랜덤 초기화 백본]")
    Xtr_r_all = torch.tensor(rand["X_train"], dtype=torch.float32)
    Xte_r     = torch.tensor(rand["X_test"],  dtype=torch.float32)
    Xva_r, Xtr_r = Xtr_r_all[val_idx], Xtr_r_all[tr_idx]

    print()
    results, hists, pred_pre = {}, {}, None
    for name, (A, B, C) in {
            "(a) ImageNet 사전학습": (Xtr, Xva, Xte),
            "(b) 랜덤 초기화":       (Xtr_r, Xva_r, Xte_r)}.items():
        t0 = time.time()
        te, va, h, pred, head = train_head(A, ytr, B, yva, C, yte)
        results[name] = dict(test_acc=round(te, 2), val_acc=round(va, 2),
                             train_s=round(time.time() - t0, 1))
        hists[name] = h
        if "사전학습" in name:
            pred_pre = pred
            torch.save(head.state_dict(), OUT / "head_pretrained.pt")
        print(f"           {name:<22} test {te:5.2f}%   val {va:5.2f}%   "
              f"헤드 학습 {time.time()-t0:.0f}초")

    gap = (results["(a) ImageNet 사전학습"]["test_acc"]
           - results["(b) 랜덤 초기화"]["test_acc"])
    print(f"\n           차이 {gap:+.2f}%p. 백본 구조도 학습 예산도 완전히 같다.")
    print( "           다른 건 '백본이 ImageNet 을 본 적이 있느냐' 하나뿐이다.")
    summary["pretrained_vs_random"] = results

    # ============================================================ #
    #  Step 4. 데이터 양 줄이기
    # ============================================================ #
    print("\n" + "-" * 74)
    print("[Step 4] 학습 데이터를 줄이면? (캐시가 있으니 이 실험이 거의 공짜다)")
    print(f"{'학습 데이터':>12}{'사전학습':>12}{'랜덤':>10}{'차이':>10}")

    sizes = [100, 500, 2000, 10000, 45000]
    part4 = {}
    for n in sizes:
        idx = torch.randperm(len(Xtr),
                             generator=torch.Generator().manual_seed(SEED))[:n]
        te_p, *_ = train_head(Xtr[idx],   ytr[idx], Xva,   yva, Xte,   yte, epochs=40)
        te_r, *_ = train_head(Xtr_r[idx], ytr[idx], Xva_r, yva, Xte_r, yte, epochs=40)
        part4[n] = dict(pretrained=round(te_p, 2), random=round(te_r, 2))
        print(f"{n:>10}장{te_p:>11.2f}%{te_r:>9.2f}%{te_p-te_r:>9.1f}%p")
    print("\n         데이터가 적을수록 사전학습의 값어치가 크다.")
    print("         100장으로도 쓸 만한 성능이 나오는 게 전이학습을 쓰는 이유다.")
    summary["data_size_sweep"] = part4

    # ============================================================ #
    #  그림
    # ============================================================ #
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for name, c in [("(a) ImageNet 사전학습", "tab:blue"),
                    ("(b) 랜덤 초기화", "tab:red")]:
        lab = "ImageNet pretrained" if "사전" in name else "random init"
        axes[0].plot(hists[name]["train"], color=c, ls="--", lw=1.5, alpha=.6)
        axes[0].plot(hists[name]["val"],   color=c, lw=2, label=lab)
    axes[0].set(xlabel="epoch", ylabel="accuracy [%]", ylim=(0, 100),
                title="(a) pretrained vs random backbone\n(dashed = train, solid = val)")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=9, loc="lower right")

    axes[1].semilogx(sizes, [part4[n]["pretrained"] for n in sizes], "o-",
                     color="tab:blue", lw=2, ms=7, label="ImageNet pretrained")
    axes[1].semilogx(sizes, [part4[n]["random"] for n in sizes], "s-",
                     color="tab:red", lw=2, ms=7, label="random init")
    axes[1].axhline(10, color="gray", ls=":", lw=1)
    axes[1].text(120, 12, "chance level (10 classes)", fontsize=8, color="gray")
    axes[1].set(xlabel="training images", ylabel="test accuracy [%]", ylim=(0, 100),
                title="(b) transfer learning wins most\nwhen data is scarce")
    axes[1].grid(alpha=.3, which="both"); axes[1].legend(fontsize=9)

    acc_cls = [100 * np.mean(pred_pre[yte.numpy() == i] == i) for i in range(10)]
    order = np.argsort(acc_cls)
    axes[2].barh([CLASSES[i] for i in order], [acc_cls[i] for i in order],
                 color="tab:blue", alpha=.85)
    axes[2].axvline(np.mean(acc_cls), color="k", ls="--", lw=1.5)
    axes[2].text(np.mean(acc_cls) + 1, 0.2, f"mean {np.mean(acc_cls):.1f}%", fontsize=9)
    axes[2].set(xlabel="accuracy [%]", xlim=(0, 100), title="(c) per-class accuracy")
    axes[2].grid(alpha=.3, axis="x")

    fig.suptitle("CIFAR-10 transfer learning: freeze the backbone, cache the features",
                 fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "cnn_transfer.png", dpi=120); plt.close(fig)

    summary["per_class_acc"] = {CLASSES[i]: round(acc_cls[i], 2) for i in range(10)}
    with open(OUT / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f"최종 test accuracy : "
          f"{results['(a) ImageNet 사전학습']['test_acc']:.2f}%  (노션 리더보드에 이 값)")
    print(f"결과 저장: {OUT}")
    print(f"feature 캐시: {SHARED}  (git 으로 동기화되니 다른 노트북에선 재추출 불필요)")


if __name__ == "__main__":
    main()
