#==========================================#
# Title:  01-001 MLP — 학습
#         구조가 다른 MLP 두 개를 같은 조건으로 MNIST 에 학습시킨다.
#         학습에는 train / validation 만 쓰고, test 는 test.py 에서만 쓴다.
# Author: Hyun Han
# Date:   2026-09-09
#
# 사용법:  python train.py      -> results/model_A.pt, model_B.pt, history.json 생성
#          python test.py       -> 저장된 가중치로 test 성능 비교
#==========================================#
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------------------------------------------------------- #
#  설정 (train.py 와 test.py 가 공유한다)
# ---------------------------------------------------------------- #
SEED        = 42
BATCH_SIZE  = 128
NUM_CLASSES = 10
EPOCHS      = 15
LR          = 1e-3
VAL_RATIO   = 1 / 6        # 60,000 -> 50,000 train + 10,000 validation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
OUT  = HERE / "results"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- #
#  두 개의 서로 다른 MLP
#    Model A : 얕고 좁게   784 -> 128 -> 10                 (은닉 1층, dropout 없음)
#    Model B : 깊고 넓게   784 -> 512 -> 512 -> 256 -> 10   (은닉 3층, dropout 0.2)
#  둘 다 fully-connected 만 쓰므로, 성능 차이는 깊이와 너비에서만 온다.
# ---------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, hidden_sizes, dropout=0.0):
        super().__init__()
        layers, in_dim = [], 784
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, NUM_CLASSES)]   # logit 출력
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, 784))


MODELS = {
    "A": dict(hidden_sizes=[128],           dropout=0.0, name="A (shallow/narrow)"),
    "B": dict(hidden_sizes=[512, 512, 256], dropout=0.2, name="B (deep/wide)"),
}


# ---------------------------------------------------------------- #
#  데이터: MNIST 를 train / validation / test 세 개로 나눈다
#  MNIST 는 train(60,000)/test(10,000) 만 주므로 train 을 한 번 더 쪼갠다.
#  시드를 고정해 두 모델이 완전히 같은 분할을 쓰게 한다.
# ---------------------------------------------------------------- #
def load_data():
    """(train_set, val_set, test_set) 을 돌려준다.

    자동 다운로드가 막힌 환경이면 MNIST 원본 4개 파일을
    <repo>/data/MNIST/raw/ 에 넣어두면 torchvision 이 그대로 읽는다.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = datasets.MNIST(DATA, train=True,  transform=tf, download=True)
    test_set   = datasets.MNIST(DATA, train=False, transform=tf, download=True)

    n_val = int(len(full_train) * VAL_RATIO)
    train_set, val_set = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED))
    return train_set, val_set, test_set


def evaluate(model, loader, criterion):
    """평균 손실과 정확도 (기울기 계산 없음)"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            correct  += (outputs.argmax(1) == labels).sum().item()
            total    += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


# ---------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    print("=" * 72)
    print(f"device: {device}")

    train_set, val_set, test_set = load_data()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"train {len(train_set)} / validation {len(val_set)} / test {len(test_set)}")
    print("학습에는 train 과 validation 만 사용한다. test 는 test.py 에서만 쓴다.")

    history, summary = {}, {}

    for key, cfg in MODELS.items():
        torch.manual_seed(SEED)                       # 같은 초기화 시드
        model = MLP(cfg["hidden_sizes"], cfg["dropout"]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== Model {cfg['name']} | {n_params:,} parameters ===")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=LR)

        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val, best_state, best_epoch = -1.0, None, -1
        t0 = time.time()

        for epoch in range(EPOCHS):
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * labels.size(0)
                correct  += (outputs.argmax(1) == labels).sum().item()
                total    += labels.size(0)

            tr_loss, tr_acc = run_loss / total, 100.0 * correct / total
            va_loss, va_acc = evaluate(model, val_loader, criterion)
            for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
                hist[k].append(v)

            # validation 이 가장 좋았던 시점의 가중치를 남긴다 (모델 선택)
            if va_acc > best_val:
                best_val, best_epoch = va_acc, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
                  f"train loss {tr_loss:.4f} acc {tr_acc:5.2f}% | "
                  f"val loss {va_loss:.4f} acc {va_acc:5.2f}%")

        torch.save(best_state, OUT / f"model_{key}.pt")
        history[key] = hist
        summary[key] = dict(name=cfg["name"], params=n_params,
                            hidden=cfg["hidden_sizes"], dropout=cfg["dropout"],
                            best_epoch=best_epoch, best_val_acc=round(best_val, 2),
                            final_train_acc=round(hist["train_acc"][-1], 2),
                            train_time_s=round(time.time() - t0, 1))
        print(f"--> best epoch {best_epoch}, val {best_val:.2f}% "
              f"-> results/model_{key}.pt 저장")

    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(OUT / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("학습 완료. 이제 test.py 를 실행해 두 모델을 비교할 것.")


if __name__ == "__main__":
    main()
