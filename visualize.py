"""可视化工具：训练后运行,自动生成 README 用图。

生成三张 PNG 到 ./assets/ 目录:
  - training_curves.png   训练 loss + train/test accuracy 曲线
  - confusion_matrix.png  测试集 10 类混淆矩阵
  - sample_predictions.png  随机 25 张测试图 + 模型预测

运行:
    python visualize.py
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CNN_case import SimpleCNN, build_dataloaders, evaluate, train_one_epoch

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# 让中文字体尽量不乱码(失败也无所谓,不影响出图)
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def plot_training_curves(history: dict, save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "o-", color="#d62728", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", color="#1f77b4", label="Train Acc")
    ax2.plot(epochs, history["test_acc"], "s-", color="#2ca02c", label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练 / 测试准确率")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path.name}")


def plot_confusion_matrix(model, loader, device, save_path: Path) -> None:
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.item(), p.item()] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵 (测试集)")

    # 在每格写上数字
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path.name}")


def plot_sample_predictions(model, loader, device, save_path: Path, n: int = 25) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    # 随机抽 n 张
    idx = random.sample(range(images.size(0)), min(n, images.size(0)))
    images = images[idx].cpu()
    labels = labels[idx].cpu()
    preds = preds[idx].cpu()

    cols = 5
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.array(axes).reshape(-1)

    for ax, img, lab, pred in zip(axes, images, labels, preds):
        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if lab.item() == pred.item() else "red"
        ax.set_title(f"真:{lab.item()} 预:{pred.item()}", color=color, fontsize=9)
        ax.axis("off")

    # 把多余的子图关掉
    for ax in axes[len(idx):]:
        ax.axis("off")

    fig.suptitle("样例预测 (绿=正确, 红=错误)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path.name}")


def main(epochs: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, test_loader = build_dataloaders()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_acc"].append(te)
        print(f"Epoch {epoch}/{epochs}  loss={tl:.4f}  train_acc={ta:.4f}  test_acc={te:.4f}")

    print(f"\n生成图片到 {ASSETS}/ :")
    plot_training_curves(history, ASSETS / "training_curves.png")
    plot_confusion_matrix(model, test_loader, device, ASSETS / "confusion_matrix.png")
    plot_sample_predictions(model, test_loader, device, ASSETS / "sample_predictions.png")
    print("全部完成 ✅")


if __name__ == "__main__":
    main()
