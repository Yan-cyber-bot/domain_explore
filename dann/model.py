import torch.nn as nn
from torchvision import models
from functions import ReverseLayerF

class DANNResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(DANNResNet50, self).__init__()
        # 使用 ResNet-50 作为特征提取器，去掉最后的全连接层
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feature = nn.Sequential(*list(backbone.children())[:-1])  # 输出维度 [batch, 2048, 1, 1]
        self.feature_dim = 2048

        # 类别分类器（用于源域分类）
        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # 域判别器（用于对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha):
        # 提取特征
        features = self.feature(x)  # shape: [batch, 2048, 1, 1]
        # 类别分类（使用原始特征）
        class_output = self.class_classifier(features)
        # 域分类（使用梯度反转后的特征）
        reversed_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        return class_output, domain_output

class STAFFResNet50(DANNResNet50):
    """
    继承 DANNResNet50，添加 MI 模块，前向返回 (class_output, domain_output, l, g)
    其中 l: 局部特征图 [batch, 2048, H, W] ，g: 全局特征 [batch, 2048, 1, 1]
    """
    def __init__(self, num_classes=4, pretrained=True, local_feat_dim=256):
        super(STAFFResNet50, self).__init__(num_classes, pretrained)
        # 重新定义 backbone 以获取中间层输出 (layer4)
        backbone_full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # 拆分为卷积部分和全局池化部分
        self.backbone_layers = nn.Sequential(*list(backbone_full.children())[:-2])  # 输出 [batch, 2048, 7, 7]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 注意：父类的 self.feature 不再使用，但保留以保持兼容性（也可删除）
        self.feature = None  # 禁用父类 feature，使用新的结构

        # MI 模块
        self.M_G = nn.Linear(num_classes, self.feature_dim, bias=False)
        self.local_proj = nn.Conv2d(self.feature_dim, local_feat_dim, kernel_size=1)
        self.g_proj_local = nn.Linear(self.feature_dim, local_feat_dim, bias=False)

    def forward(self, x, alpha):
        l = self.backbone_layers(x)          # [batch, 2048, H, W]
        g = self.global_pool(l)              # [batch, 2048, 1, 1]
        h = self.class_classifier(g)         # [batch, num_classes]
        reversed_g = ReverseLayerF.apply(g, alpha)
        domain_output = self.domain_classifier(reversed_g)
        return h, domain_output, l, g