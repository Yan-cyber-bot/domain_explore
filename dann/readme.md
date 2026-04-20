# DANN for UC Classification (ResNet-50 Backbone)

本项目基于 **Unsupervised Domain Adaptation by Backpropagation** (DANN) 论文思想，使用 PyTorch 实现域对抗神经网络，用于跨中心的溃疡性结肠炎（UC）内镜图像分类。

## 主要改进
- 将原始 CNN 骨干网络替换为预训练的 **ResNet-50**。
- 适配自定义的 `UCDataset` 数据集加载方式（支持 Excel 划分）。
- 使用现代 PyTorch 训练范式，兼容最新版库。
- 保留核心的梯度反转层（GRL），实现无监督域适应。

## 项目结构
├── functions.py # 梯度反转层实现
├── model.py # DANN 模型定义（ResNet-50）
├── uc_dataset.py # 自定义数据集类
├── transforms.py # 数据增强策略
├── main.py # 训练脚本
├── test.py # 测试脚本
├── requirements.txt # 依赖包
└── README.md
