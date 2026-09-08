#==========================================#
# Title:  01-005 XAI — 학습, AUC, CAM 분석
#
#   데이터셋: Oxford-IIIT Pet (고양이 vs 개, 이진분류)
#             002 에서 쓴 CIFAR-10 과 다른 public dataset.
#             224x224 실사진이라 CAM 이 의미있게 보이고,
#             동물 영역 정답 마스크(trimap)가 같이 들어있어서
#             'CAM 이 진짜 동물을 보는가'를 숫자로 잴 수 있다.
#
#   산출물: results/xai_auc.png        ROC 곡선 + CAM 집중도 분포
#           results/cam_heatmaps.png   CAM 히트맵 이미지
#
# Author: Hyun Han
# Date:   2026-09-09
#
# 사용법:  python train.py
#          SMOKE=1 python train.py     <- 가짜 데이터로 전 경로만 점검
#==========================================#
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_network import build, N_CLASSES

SEED   = 42
BATCH  = 64
EPOCHS = 40
LR     = 1e-3
IMG    = 224
CLASS_NAMES = ["cat", "dog"]

SMOKE       = os.environ.get("SMOKE", "0") == "1"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[1]
DATA   = ROOT / "data"
SHARED = ROOT / "shared"
OUT    = HERE / "results"
for _d in (DATA, SHARED, OUT):
    _d.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------- #
#  데이터
# ---------------------------------------------------------------- #
def get_datasets():
    """(train_ds, test_ds, seg_ds) — seg_ds 는 test 와 순서가 같은 마스크용."""
    if SMOKE:
        return None, None, None
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    tf_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(IMG),
        transforms.PILToTensor(),
    ])

    kw = dict(root=str(DATA), download=True, transform=tf_img)
    train_ds = datasets.OxfordIIITPet(split="trainval",
                                      target_types="binary-category", **kw)
    test_ds  = datasets.OxfordIIITPet(split="test",
                                      target_types="binary-category", **kw)
    seg_ds   = datasets.OxfordIIITPet(split="test", target_types="segmentation",
                                      target_transform=tf_mask, **kw)
    return train_ds, test_ds, seg_ds


def fake_batch(n, seed=0):
    """SMOKE 모드용 가짜 데이터. 이미지·라벨·trimap 마스크를 같이 만든다."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, N_CLASSES, (n,), generator=g)
    x = torch.randn(n, 3, IMG, IMG, generator=g) * 0.5
    x[y == 1, 0] += 0.8                      # dog 은 R 채널이 밝다
    x[:, :, 80:150, 80:150] += y[:, None, None, None].float()
    m = torch.full((n, 1, IMG, IMG), 2, dtype=torch.uint8)   # 2 = 배경
    m[:, :, 80:150, 80:150] = 1                              # 1 = 동물
    return x, y, m


@torch.no_grad()
def extract_gap(model, ds, tag):
    """GAP 특징 (N, 512) 을 뽑아 캐시한다. backbone 이 freeze 라 한 번이면 된다."""
    cache = SHARED / f"pets_{tag}_gap_fp16.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"[cache] {cache.name} 재사용 ({len(d['X'])}장)")
        return torch.tensor(d["X"], dtype=torch.float32), torch.tensor(d["y"]).long()

    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    fs, ys, t0 = [], [], time.time()
    model.eval()
    for i, (x, y) in enumerate(loader):
        fs.append(model.gap(x.to(device, non_blocking=True)).cpu())
        ys.append(y)
        if (i + 1) % 10 == 0:
            done = (i + 1) * BATCH
            print(f"        {tag} {done:>5}/{len(ds)}  "
                  f"{done/(time.time()-t0):.0f} img/s", end="\r")
    X = torch.cat(fs); Y = torch.cat(ys).long()
    np.savez_compressed(cache, X=X.numpy().astype(np.float16), y=Y.numpy())
    print(f"        {tag} {len(ds)}장 완료 ({time.time()-t0:.0f}초) -> {cache.name}")
    return X, Y


# ---------------------------------------------------------------- #
#  AUC — sklearn 없이도 돌아가게 직접 구현
#
#  AUC = 무작위로 고른 양성 1개의 점수가 무작위로 고른 음성 1개보다 높을 확률.
#  순위합(Mann-Whitney U)으로 정확히 계산된다.
# ---------------------------------------------------------------- #
def roc_auc(y_true, score):
    y = np.asarray(y_true).ravel(); s = np.asarray(score).ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 동점 처리 (평균 순위)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i + 1, j + 2).mean()
        i = j + 1
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(y_true, score, n_pts=200):
    y = np.asarray(y_true).ravel(); s = np.asarray(score).ravel()
    ths = np.quantile(s, np.linspace(0, 1, n_pts))
    P, N = (y == 1).sum(), (y == 0).sum()
    tpr = np.array([((s >= t) & (y == 1)).sum() / P for t in ths])
    fpr = np.array([((s >= t) & (y == 0)).sum() / N for t in ths])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


# ---------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 74)
    print(f"device: {device}"
          f"{'  (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    if SMOKE:
        print("*** SMOKE 모드: 가짜 데이터로 코드 경로만 점검한다 ***")
    summary = {"dataset": "Oxford-IIIT Pet (binary: cat vs dog)",
               "smoke": SMOKE, "device": str(device)}

    model = build(pretrained=not SMOKE, freeze=True).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델: ImageNet 사전학습 ResNet18 (freeze) + Linear(512->2)")
    print(f"      학습 대상 파라미터 {n_tr:,}개뿐 — CAM 을 쓰려면 fc 가 한 층이어야 한다")

    # ------------------------------------------------------------ #
    #  특징 추출
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    train_ds, test_ds, seg_ds = get_datasets()
    if SMOKE:
        Xtr_img, ytr_all, _   = fake_batch(256, 1)
        Xte_img, yte, Xte_msk = fake_batch(128, 2)
        with torch.no_grad():
            Xtr_all = model.gap(Xtr_img.to(device)).cpu()
            Xte     = model.gap(Xte_img.to(device)).cpu()
    else:
        print(f"[data] Oxford-IIIT Pet  trainval {len(train_ds)} / test {len(test_ds)}")
        Xtr_all, ytr_all = extract_gap(model, train_ds, "trainval")
        Xte,     yte     = extract_gap(model, test_ds,  "test")

    # trainval 을 train/val 로 나눈다
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(Xtr_all), generator=g)
    n_val = int(len(perm) * 0.15)
    va_i, tr_i = perm[:n_val], perm[n_val:]
    Xtr, ytr = Xtr_all[tr_i], ytr_all[tr_i]
    Xva, yva = Xtr_all[va_i], ytr_all[va_i]
    print(f"[data] train {len(Xtr)} / val {len(Xva)} / test {len(Xte)}   "
          f"(test 중 dog {int((yte==1).sum())}, cat {int((yte==0).sum())})")

    # ------------------------------------------------------------ #
    #  fc 한 층만 학습
    # ------------------------------------------------------------ #
    opt = torch.optim.Adam(model.fc.parameters(), lr=LR)
    lossfn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    Xva_d, yva_d = Xva.to(device), yva.to(device)

    best_va, best_state, t0 = -1.0, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossfn(model.forward_from_gap(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = (model.forward_from_gap(Xva_d).argmax(1) == yva_d).float().mean().item()*100
        if va > best_va:
            best_va = va
            best_state = {k: v.clone() for k, v in model.fc.state_dict().items()}
    model.fc.load_state_dict(best_state)
    print(f"[학습] fc 만 {EPOCHS}에폭, {time.time()-t0:.0f}초   "
          f"최고 val 정확도 {best_va:.2f}%")

    # ------------------------------------------------------------ #
    #  test 평가 — AUC 가 이번 과제의 제출 지표
    # ------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        logits = model.forward_from_gap(Xte.to(device)).cpu()
    prob_dog = torch.softmax(logits, 1)[:, 1].numpy()
    pred = logits.argmax(1).numpy()
    y = yte.numpy()

    acc = float((pred == y).mean() * 100)
    auc = roc_auc(y, prob_dog)
    fpr, tpr = roc_curve(y, prob_dog)
    print("\n" + "-" * 74)
    print(f"[test] 정확도 {acc:.2f}%     AUC {auc:.4f}   <- 노션 리더보드에 이 값")
    summary["test"] = dict(n=len(y), accuracy=round(acc, 2), auc=round(auc, 4),
                           val_acc=round(best_va, 2))

    # ------------------------------------------------------------ #
    #  CAM — 히트맵과, '진짜 동물을 보는가' 정량 평가
    # ------------------------------------------------------------ #
    print("\n" + "-" * 74)
    print("[CAM] 히트맵 생성 및 집중도 측정")

    conf = np.abs(prob_dog - 0.5)
    n_show = 8
    pick = list(np.argsort(-conf)[:4])                      # 자신있게 맞힌 것
    wrong = np.where(pred != y)[0]
    pick += list(wrong[np.argsort(-conf[wrong])][:2]) if len(wrong) >= 2 else []
    pick += list(np.argsort(conf)[:n_show - len(pick)])     # 애매한 것
    pick = pick[:n_show]

    def get_img(i):
        return Xte_img[i] if SMOKE else test_ds[i][0]

    def get_mask(i):
        return Xte_msk[i] if SMOKE else seg_ds[i][1]     # (1,224,224) 값 1/2/3

    n_total = len(yte)
    xs = torch.stack([get_img(i) for i in pick])
    heat, _ = model.cam(xs.to(device))
    heat = heat.cpu().numpy()

    # 정량 평가: CAM 에너지 중 동물 영역 안에 떨어진 비율
    energy_in, area_frac = [], []
    n_eval = min(300, n_total)
    step = max(1, n_total // n_eval)
    idxs = list(range(0, n_total, step))[:n_eval]
    for s0 in range(0, len(idxs), 32):
        chunk = idxs[s0:s0 + 32]
        xb = torch.stack([get_img(i) for i in chunk]).to(device)
        mb = torch.stack([get_mask(i) for i in chunk])[:, 0].numpy()
        hb, _ = model.cam(xb)
        hb = hb.cpu().numpy()
        mask = (mb != 2)                                 # 1=동물, 3=경계, 2=배경
        energy_in += list((hb * mask).sum((1, 2)) / (hb.sum((1, 2)) + 1e-8))
        area_frac += list(mask.mean((1, 2)))
    energy_in, area_frac = np.array(energy_in), np.array(area_frac)
    ratio = float(energy_in.mean() / area_frac.mean())
    print(f"      평가 {len(energy_in)}장")
    print(f"      CAM 에너지의 {energy_in.mean()*100:.1f}% 가 동물 영역 안에 있다")
    print(f"      동물 영역은 화면의 {area_frac.mean()*100:.1f}% 밖에 안 된다 "
          f"(무작위 히트맵이라면 이만큼만 들어간다)")
    print(f"      -> 무작위 대비 {ratio:.2f}배 집중")
    summary["cam"] = dict(n=len(energy_in),
                          energy_in_object=round(float(energy_in.mean()), 4),
                          object_area_frac=round(float(area_frac.mean()), 4),
                          ratio=round(ratio, 2))

    # ------------------------------------------------------------ #
    #  그림 1 — ROC + CAM 집중도
    # ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(fpr, tpr, color="tab:blue", lw=2.2, label=f"ResNet18 + FC  (AUC {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="random (AUC 0.5)")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                xlim=(0, 1), ylim=(0, 1.02), title="(a) ROC — cat vs dog")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=9, loc="lower right")

    if len(energy_in):
        axes[1].hist(energy_in, bins=30, color="tab:blue", alpha=.8,
                     label="CAM energy inside animal")
        axes[1].axvline(energy_in.mean(), color="tab:blue", lw=2)
        axes[1].axvline(area_frac.mean(), color="crimson", ls="--", lw=2,
                        label=f"random baseline ({area_frac.mean()*100:.0f}%)")
        axes[1].set(xlabel="fraction of CAM energy on the animal", ylabel="images",
                    title="(b) is the model actually looking at the animal?")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=.3, axis="y")
    else:
        axes[1].text(.5, .5, "SMOKE mode", ha="center")
    fig.suptitle("Oxford-IIIT Pet — classification quality and CAM localisation", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "xai_auc.png", dpi=120); plt.close(fig)

    # ------------------------------------------------------------ #
    #  그림 2 — CAM 히트맵
    # ------------------------------------------------------------ #
    fig, axes = plt.subplots(2, n_show, figsize=(2.1 * n_show, 4.9))
    for col, i in enumerate(pick):
        img = (xs[col] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        ok = pred[i] == y[i]
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true {CLASS_NAMES[y[i]]} / pred {CLASS_NAMES[pred[i]]}\n"
                               f"p(dog)={prob_dog[i]:.2f}",
                               fontsize=8, color="black" if ok else "red")
        axes[1, col].imshow(img)
        axes[1, col].imshow(heat[col], cmap="jet", alpha=.45)
        for r in (0, 1):
            axes[r, col].axis("off")
    axes[0, 0].set_ylabel("input"); axes[1, 0].set_ylabel("CAM")
    fig.suptitle("Class Activation Map — where the score came from "
                 "(left: confident, middle: wrong, right: uncertain)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "cam_heatmaps.png", dpi=130); plt.close(fig)

    torch.save(model.fc.state_dict(), OUT / "fc_head.pt")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f"AUC {auc:.4f}   정확도 {acc:.2f}%")
    print(f"결과 저장: {OUT}")
    print("  xai_auc.png       ROC 곡선 + CAM 집중도")
    print("  cam_heatmaps.png  CAM 히트맵 이미지 (제출물)")


if __name__ == "__main__":
    main()
