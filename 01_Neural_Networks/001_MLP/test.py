#==========================================#
# Title:  01-001 MLP — 테스트 및 두 모델 비교
#         train.py 가 저장한 가중치를 불러와 test 데이터셋으로 성능을 비교한다.
#         test 데이터는 여기서 딱 한 번만 쓴다.
# Author: Hyun Han
# Date:   2026-09-09
#
# 사용법:  python train.py  먼저 실행한 뒤  python test.py
#==========================================#
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# train.py 에서 모델 정의와 데이터 분할 규칙을 그대로 가져온다.
# (train.py 는 main() 을 __main__ 가드로 감쌌으므로 import 해도 학습이 돌지 않는다)
from train import MLP, MODELS, OUT, BATCH_SIZE, device, load_data, evaluate


def main():
    print("=" * 72)
    print(f"device: {device}")

    _, _, test_set = load_data()
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    print(f"test 데이터 {len(test_set)} 장으로 평가한다 (학습에는 쓰이지 않은 데이터)")

    with open(OUT / "train_summary.json") as f:
        train_summary = json.load(f)
    with open(OUT / "history.json") as f:
        history = json.load(f)

    results = {}
    for key, cfg in MODELS.items():
        ckpt = OUT / f"model_{key}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} 가 없다. train.py 를 먼저 실행할 것.")

        model = MLP(cfg["hidden_sizes"], cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        info = train_summary[key]
        results[key] = dict(info, test_acc=round(te_acc, 2), test_loss=round(te_loss, 4))
        print(f"\nModel {cfg['name']}")
        print(f"  파라미터        {info['params']:,}")
        print(f"  best epoch      {info['best_epoch']}")
        print(f"  validation acc  {info['best_val_acc']:.2f}%")
        print(f"  TEST accuracy   {te_acc:.2f}%")

        # 테스트 샘플 예측 그림
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
        fig.suptitle(f"Model {cfg['name']} — sample test predictions", fontsize=13)
        fig.tight_layout(); fig.savefig(OUT / f"predictions_{key}.png", dpi=120)
        plt.close(fig)

    # ------------------------------------------------------------ #
    #  학습 곡선
    # ------------------------------------------------------------ #
    epochs = range(1, len(history["A"]["train_loss"]) + 1)
    colors = {"A": "tab:blue", "B": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, cfg in MODELS.items():
        h, c = history[key], colors[key]
        axes[0].plot(epochs, h["train_loss"], c=c, ls="--", label=f"{cfg['name']} train")
        axes[0].plot(epochs, h["val_loss"],   c=c, ls="-",  label=f"{cfg['name']} val")
        axes[1].plot(epochs, h["train_acc"],  c=c, ls="--", label=f"{cfg['name']} train")
        axes[1].plot(epochs, h["val_acc"],    c=c, ls="-",  label=f"{cfg['name']} val")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss", yscale="log")
    axes[1].set(xlabel="epoch", ylabel="accuracy [%]", title="Accuracy", ylim=(85, 100))
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.suptitle("MNIST MLP: shallow/narrow (A) vs deep/wide (B)", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------ #
    #  최종 비교표
    # ------------------------------------------------------------ #
    with open(OUT / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'model':<20}{'params':>10}{'best ep':>9}{'val acc':>10}"
          f"{'TEST acc':>10}{'train acc':>11}")
    print("-" * 78)
    for key, r in results.items():
        print(f"{r['name']:<20}{r['params']:>10,}{r['best_epoch']:>9}"
              f"{r['best_val_acc']:>9.2f}%{r['test_acc']:>9.2f}%"
              f"{r['final_train_acc']:>10.2f}%")
    print("=" * 78)

    a, b = results["A"], results["B"]
    print(f"\nModel 1 (A) accuracy : {a['test_acc']:.2f}%")
    print(f"Model 2 (B) accuracy : {b['test_acc']:.2f}%")
    print(f"차이 {b['test_acc'] - a['test_acc']:+.2f}%p "
          f"(파라미터는 {b['params']/a['params']:.1f}배)")
    print(f"\nA 의 train-val 격차 {a['final_train_acc'] - a['best_val_acc']:+.2f}%p "
          f"-> 과적합")
    print(f"B 의 train-val 격차 {b['final_train_acc'] - b['best_val_acc']:+.2f}%p "
          f"-> dropout 이 학습 때만 켜져 val 이 더 높게 나온다")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
