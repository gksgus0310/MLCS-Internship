#==========================================#
# Title:  01-005 XAI — 네트워크 정의 및 CAM
#
#         과제 요구: 예제와 다른 public image dataset 으로 CNN 학습 후
#                   CAM 분석. 제출물은 AUC + CAM heatmap 이미지 + 코드.
#
# Author: Hyun Han
# Date:   2026-09-09
#
# CAM (Class Activation Map) 이 성립하는 조건:
#   마지막이  conv -> Global Average Pooling -> Linear  구조여야 한다.
#
#   GAP 는 (512, 7, 7) 특징맵을 채널마다 평균 내 512개 숫자로 만든다.
#       f_k = (1/49) * sum_{x,y} A_k(x,y)
#   그 다음 Linear 가 클래스 점수를 만든다.
#       s_c = sum_k w[c,k] * f_k
#
#   두 식을 합치면 합의 순서를 바꿀 수 있다.
#       s_c = (1/49) * sum_{x,y} [ sum_k w[c,k] * A_k(x,y) ]
#                                 ^^^^^^^^^^^^^^^^^^^^^^^^
#                                 이게 CAM_c(x,y)
#
#   즉 CAM 은 "이 위치가 클래스 c 점수에 얼마나 기여했는가"를 정확히 나타낸다.
#   근사가 아니라 항등식이다. GAP 를 거치며 뭉개진 공간 정보를 되살리는 것.
#
#   ResNet18 은 layer4 -> avgpool -> fc 구조라 조건을 그대로 만족한다.
#   단, fc 가 Linear 한 층이어야 한다 (은닉층을 끼우면 위 식이 깨진다).
#==========================================#
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

N_CLASSES = 2          # 0 = cat, 1 = dog


class CAMResNet18(nn.Module):
    """ImageNet 사전학습 ResNet18 + 새 Linear 한 층. CAM 을 뽑을 수 있다."""

    def __init__(self, n_classes=N_CLASSES, pretrained=True, freeze=True):
        super().__init__()
        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # avgpool 직전까지를 특징 추출부로 떼어낸다 -> 출력 (B, 512, 7, 7)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4)
        self.n_feat = 512

        if freeze:
            for p in self.features.parameters():
                p.requires_grad_(False)
            self.features.eval()

        # CAM 을 쓰려면 반드시 Linear 한 층이어야 한다 (은닉층 금지)
        self.fc = nn.Linear(self.n_feat, n_classes)

    def train(self, mode=True):
        """freeze 한 backbone 은 항상 eval 로 둔다 (BatchNorm 통계 고정)."""
        super().train(mode)
        if not any(p.requires_grad for p in self.features.parameters()):
            self.features.eval()
        return self

    # ------------------------------------------------------------ #
    def feature_map(self, x):
        """(B, 512, 7, 7) 공간 정보가 살아있는 특징맵"""
        return self.features(x)

    def gap(self, x):
        """(B, 512) GAP 를 거친 벡터 — 학습 캐시용"""
        return self.feature_map(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.gap(x))

    def forward_from_gap(self, f):
        """이미 뽑아둔 GAP 특징으로 바로 분류 (캐시 학습용)"""
        return self.fc(f)

    # ------------------------------------------------------------ #
    @torch.no_grad()
    def cam(self, x, class_idx=None, out_size=224):
        """CAM 히트맵을 돌려준다.

        반환: (heat, logits)
          heat   (B, out_size, out_size)  각 이미지별로 0~1 정규화
          logits (B, n_classes)
        class_idx 를 주지 않으면 모델이 예측한 클래스에 대한 CAM 을 뽑는다.
        """
        A = self.feature_map(x)                      # (B, 512, 7, 7)
        logits = self.fc(A.mean(dim=(2, 3)))         # (B, C)
        if class_idx is None:
            class_idx = logits.argmax(1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((len(x),), class_idx,
                                   dtype=torch.long, device=x.device)

        w = self.fc.weight[class_idx]                # (B, 512)
        cam = (A * w[:, :, None, None]).sum(1)       # (B, 7, 7)

        cam = F.relu(cam)                            # 음의 기여는 버린다
        cam = F.interpolate(cam[:, None], size=(out_size, out_size),
                            mode="bilinear", align_corners=False)[:, 0]

        # 이미지마다 0~1 로 정규화 (밝기를 서로 비교하려는 게 아니라 위치를 본다)
        B = cam.shape[0]
        mn = cam.view(B, -1).min(1).values[:, None, None]
        mx = cam.view(B, -1).max(1).values[:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam, logits


def build(pretrained=True, freeze=True, n_classes=N_CLASSES):
    return CAMResNet18(n_classes=n_classes, pretrained=pretrained, freeze=freeze)


if __name__ == "__main__":
    m = build(pretrained=False)
    x = torch.randn(3, 3, 224, 224)
    heat, logits = m.cam(x)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"feature_map {tuple(m.feature_map(x).shape)}")
    print(f"logits      {tuple(logits.shape)}")
    print(f"CAM         {tuple(heat.shape)}  범위 [{heat.min():.2f}, {heat.max():.2f}]")
    print(f"파라미터    학습 {n_tr:,} / 전체 {n_all:,}  (backbone freeze)")
