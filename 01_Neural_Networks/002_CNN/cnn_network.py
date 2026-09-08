#==========================================#
# Title:  01-002 CNN & Classification — 네트워크 정의
#
#         과제 요구사항(노션):
#           - CNN 아키텍처를 직접 설계하지 말고 잘 만들어진 것을 가져올 것
#           - 뒷부분 fully-connected 만 바꿔 CIFAR-10 을 분류할 것
#           - ImageNet 사전학습 가중치를 초기값으로 쓸 것
#           - CNN 층은 freeze 해도 좋다 (학습 시간 단축)
#
#         freeze 하면 backbone 을 통과하는 forward 가 매 에폭 똑같다.
#         그러니 딱 한 번만 통과시켜 512차원 feature 를 저장해두고,
#         그 다음부터는 헤드만 학습한다. 실험 속도가 100배쯤 빨라진다.
#
# Author: Hyun Han
# Date:   2026-09-09
#
# 이 파일은 정의만 한다. 실행은 train.py 에서 한다.
#==========================================#
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------- #
#  공통 설정 (train.py 가 그대로 가져다 쓴다)
# ---------------------------------------------------------------- #
SEED    = 42
BATCH   = 256
FEATURE_DIM = 512                      # ResNet18 의 avgpool 출력 차원
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[1]
DATA   = ROOT / "data"
SHARED = ROOT / "shared"
OUT    = HERE / "results"
for _d in (DATA, SHARED, OUT):
    _d.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Windows 는 DataLoader 워커를 spawn 으로 띄운다 (fork 가 없다).
# spawn 은 메인 모듈을 다시 import 하므로 train.py 쪽에
# if __name__ == "__main__": 가드가 반드시 있어야 한다.
# 그래도 문제가 나면  $env:NUM_WORKERS=0  으로 끄면 된다 (느리지만 확실).
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))

# ImageNet 사전학습 모델이 기대하는 전처리 (224x224, ImageNet 평균/표준편차)
PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------- #
#  1. Backbone — 직접 설계하지 않고 torchvision 의 ResNet18 을 가져온다
# ---------------------------------------------------------------- #
def get_backbone(pretrained=True, seed=SEED):
    """마지막 분류층을 떼어낸 ResNet18 을 돌려준다 (출력 512차원).

    pretrained=True  -> ImageNet 사전학습 가중치
    pretrained=False -> 랜덤 초기화 (비교 실험용)
    어느 쪽이든 freeze(eval + no_grad) 해서 feature 추출기로만 쓴다.
    """
    if pretrained:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        torch.manual_seed(seed)
        net = models.resnet18(weights=None)
    net.fc = nn.Identity()               # 분류층 제거 -> 512차원 feature 출력
    for p in net.parameters():           # freeze
        p.requires_grad_(False)
    return net.to(device).eval()


# ---------------------------------------------------------------- #
#  2. Head — 우리가 바꿔 끼우는 fully-connected 부분
# ---------------------------------------------------------------- #
class ClassifierHead(nn.Module):
    """512 -> 256 -> 10. 과제에서 '뒷부분 FC 만 바꾼다'에 해당하는 부분."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=256, n_cls=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


class TransferNet(nn.Module):
    """backbone + head 를 하나로 묶은 최종 분류기.

    학습은 캐시된 feature 로 head 만 돌리지만, 제출용으로 '전체 모델'
    형태도 있어야 하므로 여기에 정의해 둔다. 이미지 -> 클래스.
    """

    def __init__(self, pretrained=True, head=None):
        super().__init__()
        self.backbone = get_backbone(pretrained)
        self.head = head if head is not None else ClassifierHead()

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)


# ---------------------------------------------------------------- #
#  3. feature 추출 — backbone 이 freeze 이므로 딱 한 번만 돌린다
# ---------------------------------------------------------------- #
def extract_features(backbone, cache_path, want_labels=True, tag=""):
    """CIFAR-10 전체를 backbone 에 한 번 통과시켜 npz 로 저장하고 dict 로 돌려준다.

    fp16 으로 저장한다. 50,000 x 512 fp32 는 97.7MB 라 GitHub 경고선(50MiB)을
    넘지만, fp16 이면 48.8MB 로 들어간다. 정확도 차이는 없다.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[cache] {cache_path.name} 발견 "
              f"({cache_path.stat().st_size/1024**2:.1f} MB) — 추출 건너뜀")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    print(f"[cache] {cache_path.name} 없음 — 지금 추출한다{tag} (최초 1회)")
    out = {}
    for name, is_train in [("train", True), ("test", False)]:
        ds = datasets.CIFAR10(DATA, train=is_train,
                              transform=PREPROCESS, download=True)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=(device.type == "cuda"))
        feats, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                feats.append(backbone(x.to(device, non_blocking=True)).cpu())
                labels.append(y)
                if (i + 1) % 20 == 0:
                    done = (i + 1) * BATCH
                    print(f"        {name} {done:>6}/{len(ds)}  "
                          f"{done/(time.time()-t0):.0f} img/s", end="\r")
        out[f"X_{name}"] = torch.cat(feats).numpy().astype(np.float16)
        if want_labels:
            out[f"y_{name}"] = torch.cat(labels).numpy().astype(np.int16)
        print(f"        {name} {len(ds)} 장 완료 ({time.time()-t0:.0f}초)          ")

    np.savez_compressed(cache_path, **out)
    print(f"[cache] 저장: {cache_path.name} "
          f"({cache_path.stat().st_size/1024**2:.1f} MB)")
    return out
