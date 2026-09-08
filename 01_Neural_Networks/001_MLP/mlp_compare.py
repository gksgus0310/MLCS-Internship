#==========================================#
# Title:  MLP comparison on MNIST (Task 001)
#         Two MLPs with different depth/width,
#         trained on the same train/val split.
# Author: Hyun Han
# Date:   2026-08-25
#==========================================#
import json
import platform
import socket
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")           # headless environment
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Hyper-parameters (identical for both models on purpose:
# the ONLY difference we study is the network architecture)
# ------------------------------------------------------------------ #
SEED         = 42
BATCH_SIZE   = 128
NUM_CLASSES  = 10
EPOCHS       = 15
LR           = 1e-3
VAL_RATIO    = 1 / 6            # 60,000 -> 50,000 train + 10,000 val

# 경로는 항상 이 파일 위치 기준으로 잡는다 (노트북이 바뀌어도 그대로 동작).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                  # 레포 최상단
DATA = ROOT / "data"                    # 데이터는 레포 전체가 공유 (.gitignore 처리됨)
OUT  = HERE / "results"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MACHINE = dict(host=socket.gethostname(), os=f"{platform.system()} {platform.release()}",
               torch=torch.__version__, device=str(device),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print(f"machine: {MACHINE['host']} | device: {device} | gpu: {MACHINE['gpu']}")

"""
Step 1: load datasets and split them into THREE sets
 MNIST ships with only train(60,000) / test(10,000).
 The task requires a validation set, so the official training set is
 split once (fixed seed) into train / validation.  The official test
 set is never touched until the very last evaluation.

 NOTE: 자동 다운로드가 막힌 환경이면 MNIST 원본 4개 파일을
 <repo>/data/MNIST/raw/ 에 넣어두면 torchvision이 그대로 읽는다.
 (scripts/download_mnist.py 참고)
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

full_train = datasets.MNIST(root=DATA, train=True,  transform=transform, download=True)
test_set   = datasets.MNIST(root=DATA, train=False, transform=transform, download=True)

n_val   = int(len(full_train) * VAL_RATIO)
n_train = len(full_train) - n_val
train_set, val_set = random_split(
    full_train, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)   # same split for both models
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

print(f"train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")

"""
Step 2: define the two MLPs
 Model A : shallow & narrow  784 -> 128 -> 10                (1 hidden layer)
 Model B : deep & wide       784 -> 512 -> 512 -> 256 -> 10  (3 hidden layers + dropout)
 Both are pure fully-connected networks, so the comparison isolates
 the effect of depth and layer width.
"""
class MLP(nn.Module):
    def __init__(self, hidden_sizes, dropout=0.0):
        super().__init__()
        layers, in_dim = [], 784
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, NUM_CLASSES)]   # logits (CrossEntropyLoss applies softmax)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, 784))


MODELS = {
    "A (shallow/narrow)": dict(hidden_sizes=[128],           dropout=0.0),
    "B (deep/wide)":      dict(hidden_sizes=[512, 512, 256], dropout=0.2),
}


# ------------------------------------------------------------------ #
def evaluate(model, loader, criterion):
    """average loss and accuracy over a loader (no gradient)"""
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


"""
Step 3: train both models
 CrossEntropyLoss + RMSprop, exactly the same budget for both.
 During training ONLY the train and validation sets are used.
 The parameters with the best validation accuracy are kept
 (early-stopping style model selection) -- this is what the
 validation set is for.
"""
history, summary = {}, {}

for name, cfg in MODELS.items():
    torch.manual_seed(SEED)                       # same initialisation seed
    model = MLP(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} | {n_params:,} parameters ===")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state, best_epoch = -1.0, None, -1
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

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc);   hist["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{str(epoch+1).zfill(2)}/{EPOCHS}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:5.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc:5.2f}%")

    train_time = time.time() - t0

    """
    Step 4: test the selected model
     The test set is used exactly once, on the best-validation weights.
    """
    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"--> best epoch {best_epoch} | val {best_val_acc:.2f}% | TEST {te_acc:.2f}%")

    history[name] = hist
    summary[name] = dict(params=n_params, best_epoch=best_epoch,
                         best_val_acc=best_val_acc, test_acc=te_acc,
                         test_loss=te_loss, final_train_acc=hist["train_acc"][-1],
                         train_time_s=round(train_time, 1),
                         hidden=cfg["hidden_sizes"], dropout=cfg["dropout"])
    torch.save(best_state, OUT / f"model_{name[0]}.pt")

    # sample predictions of this model
    model.eval()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze().numpy() * 0.5 + 0.5, cmap="gray")
        ok = preds[i].item() == labels[i].item()
        ax.set_title(f"True {labels[i].item()} / Pred {preds[i].item()}",
                     fontsize=10, color="black" if ok else "red")
        ax.axis("off")
    fig.suptitle(f"Model {name} - sample test predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"predictions_{name[0]}.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Learning curves: train vs validation for both models
# ------------------------------------------------------------------ #
epochs_axis = range(1, EPOCHS + 1)
colors = {"A (shallow/narrow)": "tab:blue", "B (deep/wide)": "tab:red"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, hist in history.items():
    c = colors[name]
    axes[0].plot(epochs_axis, hist["train_loss"], c=c, ls="--", label=f"{name} train")
    axes[0].plot(epochs_axis, hist["val_loss"],   c=c, ls="-",  label=f"{name} val")
    axes[1].plot(epochs_axis, hist["train_acc"],  c=c, ls="--", label=f"{name} train")
    axes[1].plot(epochs_axis, hist["val_acc"],    c=c, ls="-",  label=f"{name} val")

axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Loss"); axes[0].set_yscale("log"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=9)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy [%]")
axes[1].set_title("Accuracy"); axes[1].set_ylim(85, 100); axes[1].grid(alpha=.3); axes[1].legend(fontsize=9)
fig.suptitle("MNIST MLP: shallow/narrow (A) vs deep/wide (B)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "learning_curves.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ #
# Summary table
# ------------------------------------------------------------------ #
# 어느 노트북에서 뽑은 결과인지 같이 남긴다 (CPU/GPU에 따라 수치가 미세하게 다르다)
with open(OUT / "summary.json", "w") as f:
    json.dump({"machine": MACHINE, "results": summary}, f, indent=2)
with open(OUT / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 78)
print(f"{'model':<20}{'params':>10}{'best ep':>9}{'val acc':>10}{'test acc':>10}{'train s':>10}")
print("-" * 78)
for name, s in summary.items():
    print(f"{name:<20}{s['params']:>10,}{s['best_epoch']:>9}"
          f"{s['best_val_acc']:>9.2f}%{s['test_acc']:>9.2f}%{s['train_time_s']:>10.1f}")
print("=" * 78)
print(f"results saved to {OUT}")
