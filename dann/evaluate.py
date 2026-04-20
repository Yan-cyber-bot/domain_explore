import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
import pandas as pd
import sys

sys.path.append('/root/autodl-tmp/domain_explore')  # 根据实际路径调整
from src.datasets.uc_dataset import UCDataset
from src.datasets.transforms import get_val_transforms
from model import DANNResNet50, STAFFResNet50  # 导入两个模型类


def evaluate(model, dataloader, device, num_classes=4):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, alpha=0)  # 测试时 alpha=0
            # 通用处理：第一个返回值必定是分类结果
            class_output = outputs[0]
            probs = torch.exp(class_output).cpu().numpy()
            preds = class_output.data.max(1)[1].cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes))
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        auc_per_class = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    except ValueError:
        auc_macro = auc_weighted = np.nan
        auc_per_class = [np.nan] * num_classes

    metrics = {
        'accuracy': acc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'auc_per_class': auc_per_class,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    return metrics, all_labels, all_probs, all_preds


def save_results(metrics, all_labels, all_probs, all_preds, metrics_path, preds_path, class_names):
    # 构建汇总指标 DataFrame
    metrics_data = {
        'Metric': [
            'Accuracy',
            'Macro Precision', 'Macro Recall', 'Macro F1', 'Macro AUC',
            'Weighted Precision', 'Weighted Recall', 'Weighted F1', 'Weighted AUC'
        ],
        'Value': [
            metrics['accuracy'],
            metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['auc_macro'],
            metrics['weighted_precision'], metrics['weighted_recall'], metrics['weighted_f1'], metrics['auc_weighted']
        ]
    }
    for i, name in enumerate(class_names):
        metrics_data['Metric'].extend([
            f'{name} Precision', f'{name} Recall', f'{name} F1', f'{name} AUC', f'{name} Support'
        ])
        metrics_data['Value'].extend([
            metrics['precision_per_class'][i],
            metrics['recall_per_class'][i],
            metrics['f1_per_class'][i],
            metrics['auc_per_class'][i],
            metrics['support_per_class'][i]
        ])

    metrics_df = pd.DataFrame(metrics_data)
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=[f'True_{n}' for n in class_names],
        columns=[f'Pred_{n}' for n in class_names]
    )

    if metrics_path.endswith('.csv'):
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        cm_path = metrics_path.replace('.csv', '_confusion.csv')
        cm_df.to_csv(cm_path, encoding='utf-8-sig')
    else:
        with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    print(f'Metrics saved to {metrics_path}')

    if preds_path is not None:
        preds_df = pd.DataFrame({
            'True_Label': all_labels,
            'Pred_Label': all_preds,
        })
        for i, name in enumerate(class_names):
            preds_df[f'Prob_{name}'] = all_probs[:, i]

        if preds_path.endswith('.csv'):
            preds_df.to_csv(preds_path, index=False, encoding='utf-8-sig')
        else:
            preds_df.to_excel(preds_path, index=False, engine='openpyxl')
        print(f'Predictions saved to {preds_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate DANN/STAFF model on UC datasets')
    parser.add_argument('--model_type', type=str, default='dann', choices=['dann', 'staff'],
                        help='Type of model (dann or staff)')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--excel_path', type=str, default='../data/UC_all_divided_data2.xlsx',
                        help='Path to the Excel file with data splits')
    parser.add_argument('--base_dir', type=str, default='../data',
                        help='Base directory for images')
    parser.add_argument('--dataset', type=str, choices=['邵逸夫UC', '大坪UC'], required=True,
                        help='Which dataset to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split: "train", "test", or "all"')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--save_metrics', type=str, default=None,
                        help='Path to save summary metrics (CSV or Excel)')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Path to save per-sample predictions (CSV or Excel)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    transform = get_val_transforms(img_size=args.image_size)
    split = None if args.split.lower() == 'all' else args.split

    dataset = UCDataset(
        excel_path=args.excel_path,
        base_dir=args.base_dir,
        transform=transform,
        split=split,
        source=args.dataset,
        label_filter=None,
        return_info=False
    )
    print(f'Loaded {len(dataset)} samples from {args.dataset} (split: {args.split})')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 根据模型类型初始化模型
    if args.model_type == 'dann':
        model = DANNResNet50(num_classes=args.num_classes, pretrained=False)
    elif args.model_type == 'staff':
        model = STAFFResNet50(num_classes=args.num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f'Loaded {args.model_type} model from {args.model_path}')

    metrics, all_labels, all_probs, all_preds = evaluate(model, dataloader, device, num_classes=args.num_classes)

    class_names = ['0', '1', '2', '3']  # 可按需修改为实际类别名
    print('\n' + '=' * 50)
    print(f'Evaluation on {args.dataset} ({args.split} split)')
    print('=' * 50)
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print('\n--- Per-class Metrics ---')
    for i, name in enumerate(class_names):
        print(f'Class {name}:')
        print(f'  Precision: {metrics["precision_per_class"][i]:.4f}')
        print(f'  Recall:    {metrics["recall_per_class"][i]:.4f}')
        print(f'  F1-score:  {metrics["f1_per_class"][i]:.4f}')
        print(f'  Support:   {metrics["support_per_class"][i]}')
        if not np.isnan(metrics["auc_per_class"][i]):
            print(f'  AUC:       {metrics["auc_per_class"][i]:.4f}')
        else:
            print('  AUC:       N/A')
        print()

    print('--- Macro Average ---')
    print(f'Precision: {metrics["macro_precision"]:.4f}')
    print(f'Recall:    {metrics["macro_recall"]:.4f}')
    print(f'F1-score:  {metrics["macro_f1"]:.4f}')
    print(f'AUC:       {metrics["auc_macro"]:.4f}' if not np.isnan(metrics["auc_macro"]) else 'AUC: N/A')

    print('\n--- Weighted Average ---')
    print(f'Precision: {metrics["weighted_precision"]:.4f}')
    print(f'Recall:    {metrics["weighted_recall"]:.4f}')
    print(f'F1-score:  {metrics["weighted_f1"]:.4f}')
    print(f'AUC:       {metrics["auc_weighted"]:.4f}' if not np.isnan(metrics["auc_weighted"]) else 'AUC: N/A')

    print('\n--- Confusion Matrix ---')
    print(pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names))

    if args.save_metrics:
        save_results(
            metrics, all_labels, all_probs, all_preds,
            args.save_metrics, args.save_predictions,
            class_names=class_names
        )


if __name__ == '__main__':
    main()