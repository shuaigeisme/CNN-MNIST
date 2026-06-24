# CNN-MNIST · 简单的卷积神经网络手写数字识别

一个用 **PyTorch** 实现的轻量级卷积神经网络(CNN),用于 MNIST 手写数字识别。
模型结构:两层卷积(带 BN + ReLU + MaxPool)+ 两层全连接(含 Dropout)。

## ✨ 特性

- 简洁清晰的 `SimpleCNN` 实现,代码注释详尽
- 自动检测并使用 GPU(`cuda` 可用时)
- 训练/测试一站式,单文件即可运行
- 数据复用本地 `data/MNIST/raw/`,避免重复下载

## 🧱 模型结构

```
输入 (1, 28, 28)
  └─ Conv2d(1→32, 3x3, padding=1) + BN + ReLU + MaxPool(2x2)   → (32, 14, 14)
  └─ Conv2d(32→64, 3x3, padding=1) + BN + ReLU + MaxPool(2x2)  → (64, 7, 7)
  └─ Flatten → Linear(64*7*7 → 128) + ReLU + Dropout(0.5)
  └─ Linear(128 → 10)                                           → logits
```

## 📦 环境依赖

```bash
pip install torch torchvision
```

## 🚀 运行

```bash
python CNN_case.py
```

首次运行会自动把 MNIST 下载到 `./data/` 目录(本仓库已用 `.gitignore` 排除)。

输出示例:
```
使用设备: cuda
Epoch 1/5  loss=0.2143  train_acc=0.9350  test_acc=0.9812
Epoch 2/5  loss=0.0721  train_acc=0.9778  test_acc=0.9855
...
```

## 🧪 复现结果(默认超参,5 epochs)

| 指标 | 值 |
|---|---|
| 训练集准确率 | ~99% |
| 测试集准确率 | ~99% |

## 📁 目录结构

```
.
├── CNN_case.py        # 模型定义 + 训练/评估入口
├── .gitignore         # 忽略 data/、__pycache__/、*.pth 等
└── README.md          # 本文件
```

## 📝 License

MIT
