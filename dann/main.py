import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

sys.path.append('/root/autodl-tmp/domain_explore')
from src.datasets.uc_dataset import UCDataset
from src.datasets.transforms import get_train_transforms
from model import DANNResNet50, STAFFResNet50
from losses import compute_mi_loss_global, compute_mi_loss_local
from test import test


def main():
    parser = argparse.ArgumentParser(description='DANN / STAFF Training')
    parser.add_argument('--model_type', type=str, default='dann', choices=['dann', 'staff'],
                        help='Type of model to use (dann or staff)')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for global MI loss (only for STAFF)')
    parser.add_argument('--gamma', type=float, default=0.05, help='Weight for local MI loss (only for STAFF)')
    parser.add_argument('--train_root', type=str, default='../data', help='Root directory of images')
    parser.add_argument('--excel_path', type=str, default='../data/UC_all_divided_data2.xlsx', help='Path to Excel file')
    parser.add_argument('--source_name', type=str, default='邵逸夫UC', help='Source domain name')
    parser.add_argument('--target_name', type=str, default='大坪UC', help='Target domain name')
    parser.add_argument('--model_root', type=str, default='models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    args = parser.parse_args()

    # 创建模型保存目录
    os.makedirs(args.model_root, exist_ok=True)

    # 随机种子设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    # -------------------- 数据加载 --------------------
    transform_train = get_train_transforms(img_size=args.image_size)

    dataset_source = UCDataset(
        excel_path=args.excel_path,
        base_dir=args.train_root,
        transform=transform_train,
        split='train',
        source=args.source_name,
        label_filter=None,
        return_info=False
    )

    dataset_target = UCDataset(
        excel_path=args.excel_path,
        base_dir=args.train_root,
        transform=transform_train,
        split=None,   # 使用全部目标域数据
        source=args.target_name,
        label_filter=None,
        return_info=False
    )

    dataloader_source = DataLoader(dataset_source, batch_size=args.batch_size, shuffle=True,
                                   num_workers=4, drop_last=True)
    dataloader_target = DataLoader(dataset_target, batch_size=args.batch_size, shuffle=True,
                                   num_workers=4, drop_last=True)

    # -------------------- 模型初始化 --------------------
    if args.model_type == 'dann':
        model = DANNResNet50(num_classes=args.num_classes, pretrained=True)
    elif args.model_type == 'staff':
        model = STAFFResNet50(num_classes=args.num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if cuda:
        model = model.cuda()

    # 损失函数
    loss_class = nn.NLLLoss()
    loss_domain = nn.NLLLoss()

    # 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=1e-6)

    # -------------------- 训练循环 --------------------
    best_acc_s = 0.0
    for epoch in range(args.n_epoch):
        model.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / (args.n_epoch * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # ---------- 源域数据 ----------
            s_img, s_label = next(data_source_iter)
            if cuda:
                s_img, s_label = s_img.cuda(), s_label.cuda()

            model.zero_grad()
            batch_size_s = s_img.size(0)
            domain_label_s = torch.zeros(batch_size_s, dtype=torch.long, device=s_img.device)

            if args.model_type == 'dann':
                class_output, domain_output_s = model(s_img, alpha)
                mi_loss = 0.0
            else:  # STAFF 模型返回更多输出
                class_output, domain_output_s, l_s, g_s = model(s_img, alpha)
                g_flat = g_s.squeeze(-1).squeeze(-1)
                loss_mi_global = compute_mi_loss_global(g_flat, class_output, model.M_G)
                loss_mi_local = compute_mi_loss_local(g_s, l_s, model.local_proj, model.g_proj_local)
                mi_loss = args.beta * loss_mi_global + args.gamma * loss_mi_local

            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output_s, domain_label_s)

            # ---------- 目标域数据 ----------
            t_img, _ = next(data_target_iter)
            if cuda:
                t_img = t_img.cuda()
            batch_size_t = t_img.size(0)
            domain_label_t = torch.ones(batch_size_t, dtype=torch.long, device=t_img.device)

            if args.model_type == 'dann':
                _, domain_output_t = model(t_img, alpha)
            else:
                _, domain_output_t, _, _ = model(t_img, alpha)

            err_t_domain = loss_domain(domain_output_t, domain_label_t)

            # ---------- 总损失与反向传播 ----------
            total_err = err_s_label + err_s_domain + err_t_domain + mi_loss
            total_err.backward()
            optimizer.step()

            # 打印进度
            sys.stdout.write(f'\r Epoch: {epoch:3d} [{i+1:4d}/{len_dataloader:4d}] '
                             f'loss_cls: {err_s_label.item():.4f} '
                             f'loss_dom_s: {err_s_domain.item():.4f} '
                             f'loss_dom_t: {err_t_domain.item():.4f}')
            if args.model_type == 'staff':
                sys.stdout.write(f' loss_mi: {mi_loss.item():.4f}')
            sys.stdout.flush()

        scheduler.step()
        print()

        # 测试
        acc_s = test(args.source_name, model, epoch='current')
        acc_t = test(args.target_name, model, epoch='current')
        print(f'Source ({args.source_name}) accuracy: {acc_s:.4f}')
        print(f'Target ({args.target_name}) accuracy: {acc_t:.4f}')

        # 保存最佳模型
        if acc_s > best_acc_s:
            best_acc_s = acc_s
            torch.save(model.state_dict(), os.path.join(args.model_root, 'best_model.pth'))
            print(f'Best model saved with source accuracy: {best_acc_s:.4f}')

        # 保存最近模型
        torch.save(model.state_dict(), os.path.join(args.model_root, 'latest_model.pth'))

    print('Training finished.')


if __name__ == '__main__':
    main()