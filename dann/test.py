import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('/root/autodl-tmp/domain_explore')
from src.datasets.uc_dataset import UCDataset
from src.datasets.transforms import get_val_transforms

def test(dataset_name, model=None, epoch='best'):
    """
    测试指定数据集的准确率。
    dataset_name: '大坪UC' 或 '邵逸夫UC'
    model: 模型实例，若为 None 则从文件加载
    epoch: 'best' 或 'latest' 或 'current' (用于区分模型文件)
    """
    # 配置
    model_root = 'mi_models'
    excel_path = '../data/UC_all_divided_data2.xlsx'   # 根据实际情况修改
    base_dir = '../data'
    image_size = 224
    batch_size = 32
    cuda = torch.cuda.is_available()

    # 数据预处理
    transform_val = get_val_transforms(img_size=image_size)

    dataset = UCDataset(
        excel_path=excel_path,
        base_dir=base_dir,
        transform=transform_val,
        split='test',                # 或 'test'，按需调整
        source=dataset_name,
        label_filter=None,
        return_info=False
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载模型（如果未提供）
    if model is None:
        # 这里需要导入模型类，可根据需要动态导入
        from model import DANNResNet50
        model = DANNResNet50(num_classes=4, pretrained=False)
        if epoch == 'best':
            model_path = os.path.join(model_root, 'best_model.pth')
        elif epoch == 'latest':
            model_path = os.path.join(model_root, 'latest_model.pth')
        else:
            model_path = os.path.join(model_root, 'current_model.pth')
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f'Loaded model from {model_path}')

    if cuda:
        model = model.cuda()

    model.eval()

    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images, alpha=0)
            # 通用处理：第一个输出必定是分类结果
            class_output = outputs[0]
            pred = class_output.data.max(1)[1]
            n_correct += pred.eq(labels.data).cpu().sum().item()
            n_total += labels.size(0)

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy

if __name__ == '__main__':
    # 简单测试示例
    acc_daping = test('大坪UC')
    acc_shaoyifu = test('邵逸夫UC')
    print(f'Daping accuracy: {acc_daping:.4f}')
    print(f'Shaoyifu accuracy: {acc_shaoyifu:.4f}')