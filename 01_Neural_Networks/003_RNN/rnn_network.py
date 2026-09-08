#==========================================#
# Title:  01-003 RNN & Regression — 네트워크 정의
#
#         과제 요구: LSTM 으로 주가 예측, Many-to-Many 모델 포함
#
# Author: Hyun Han
# Date:   2026-09-09
#
# LSTM 을 쓰는 이유:
#   일반 RNN 은 시점을 거슬러 올라가며 같은 가중치를 계속 곱한다.
#   그 값이 1보다 작으면 기울기가 지수적으로 0으로 죽고(vanishing),
#   1보다 크면 폭발한다. 60스텝을 거슬러 가면 사실상 학습이 안 된다.
#   LSTM 은 cell state 라는 '덧셈으로만 갱신되는 통로'를 따로 둬서
#   기울기가 그 통로를 따라 감쇠 없이 흐르게 한다. 게이트 세 개가
#   그 통로에 무엇을 넣고(input) 무엇을 지우고(forget) 무엇을 꺼낼지(output)
#   결정한다.
#==========================================#
import torch
from torch import nn

HIDDEN  = 64
LAYERS  = 2
DROPOUT = 0.2


class LSTMManyToOne(nn.Module):
    """과거 SEQ 일 -> 다음 1일.

    LSTM 을 끝까지 돌린 뒤 '마지막 시점의 은닉상태' 하나만 꺼내 FC 에 넣는다.
    중간 시점의 출력은 버린다.
    """

    def __init__(self, n_feat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                 # x: (B, T, F)
        out, _ = self.lstm(x)             # out: (B, T, H)
        return self.head(out[:, -1, :])   # (B, 1)  <- 마지막 시점만


class LSTMManyToMany(nn.Module):
    """과거 SEQ 일 -> 앞으로 HORIZON 일 (다중 스텝 예측).

    many-to-many 의 첫 번째 형태. 마지막 은닉상태 하나에서
    H 개의 값을 한 번에 뽑는다 (direct multi-step).
    한 스텝씩 자기 예측을 다시 입력으로 넣는 재귀 방식보다
    오차가 누적되지 않아 안정적이다.
    """

    def __init__(self, n_feat, horizon=5, hidden=HIDDEN,
                 layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # (B, H)


class LSTMSeqToSeq(nn.Module):
    """과거 SEQ 일 -> 각 시점의 '다음날' SEQ 개 (aligned many-to-many).

    many-to-many 의 두 번째 형태. 모든 시점의 은닉상태에 같은 FC 를 적용한다.
    출력 길이가 입력 길이와 같다.

    이렇게 하면 한 윈도우에서 기울기 신호가 1개가 아니라 T개 나온다.
    데이터가 적을 때 학습이 훨씬 안정적이다. 대신 앞쪽 시점은 참고할
    과거가 몇 개 없는 상태로 맞혀야 해서, 평가는 마지막 시점만 쓴다.
    """

    def __init__(self, n_feat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)             # (B, T, H)
        return self.head(out).squeeze(-1) # (B, T)  <- 모든 시점


def build(kind, n_feat, horizon=5):
    return {"m2o":         lambda: LSTMManyToOne(n_feat),
            "m2m":         lambda: LSTMManyToMany(n_feat, horizon),
            "m2m_aligned": lambda: LSTMSeqToSeq(n_feat)}[kind]()


if __name__ == "__main__":
    x = torch.randn(4, 60, 6)
    for k in ("m2o", "m2m", "m2m_aligned"):
        m = build(k, 6)
        n = sum(p.numel() for p in m.parameters())
        print(f"{k:<12} 출력 {tuple(m(x).shape)}   파라미터 {n:,}")
