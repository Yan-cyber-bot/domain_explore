# losses.py
import torch
import torch.nn as nn

def compute_mi_loss_global(g, h, M_G):
    """
    计算全局 MI 损失 (JSD 版本)
    g: [batch, dim] 全局特征（已 flatten）
    h: [batch, num_classes] 分类预测（LogSoftmax输出）
    M_G: nn.Linear(num_classes, dim) 投影层
    """
    batch_size = g.size(0)
    h_proj = M_G(h)  # [batch, dim]
    pos_score = torch.sum(g * h_proj, dim=1)

    shuffle_idx = torch.randperm(batch_size)
    h_proj_shuffled = h_proj[shuffle_idx]
    neg_score = torch.sum(g * h_proj_shuffled, dim=1)

    loss_pos = nn.BCEWithLogitsLoss()(pos_score, torch.ones_like(pos_score))
    loss_neg = nn.BCEWithLogitsLoss()(neg_score, torch.zeros_like(neg_score))
    return (loss_pos + loss_neg) / 2.0


def compute_mi_loss_local(g, l, local_proj, g_proj_local):
    """
    计算局部 MI 损失 (JSD 版本)
    g: [batch, feature_dim, 1, 1] 全局特征
    l: [batch, C, H, W] 局部特征图
    local_proj: nn.Conv2d(C, local_feat_dim, 1)
    g_proj_local: nn.Linear(feature_dim, local_feat_dim)
    """
    batch_size, C, H, W = l.shape
    l_proj = local_proj(l)  # [batch, local_feat_dim, H, W]

    g_flat = g.squeeze(-1).squeeze(-1)  # [batch, feature_dim]
    g_proj = g_proj_local(g_flat)       # [batch, local_feat_dim]
    g_proj = g_proj.view(batch_size, -1, 1, 1)

    scores = torch.sum(l_proj * g_proj, dim=1)  # [batch, H, W]
    pos_score = scores.view(batch_size, -1).mean(dim=1)

    shuffle_idx = torch.randperm(batch_size)
    g_proj_shuffled = g_proj[shuffle_idx]
    scores_neg = torch.sum(l_proj * g_proj_shuffled, dim=1).view(batch_size, -1).mean(dim=1)

    loss_pos = nn.BCEWithLogitsLoss()(pos_score, torch.ones_like(pos_score))
    loss_neg = nn.BCEWithLogitsLoss()(scores_neg, torch.zeros_like(scores_neg))
    return (loss_pos + loss_neg) / 2.0