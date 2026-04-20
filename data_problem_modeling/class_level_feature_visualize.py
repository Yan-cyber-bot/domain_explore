import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from tqdm import tqdm

import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import umap
import sys
sys.path.append('/root/autodl-tmp/domain_explore/')
from src.datasets import UCDataset
from src.datasets import get_val_transforms
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '../data'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
REDUCE_METHOD = 'umap'
RANDOM_SEED = 42

dataset = UCDataset(excel_path=os.path.join(DATA_DIR, 'UC_all_divided_data.xlsx'), base_dir=DATA_DIR, transform=get_val_transforms(IMG_SIZE),
                    source=['大坪UC', '邵逸夫UC'], return_info=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(DEVICE)
model.eval()

domain_id = {'大坪UC': 0, '邵逸夫UC': 1}
features = []
labels = []
sources = []

print("Extracting features...")
with torch.no_grad():
    for images, batch_labels, batch_infos in tqdm(dataloader):
        images = images.to(DEVICE)
        feats = model(images).squeeze(-1).squeeze(-1)
        features.append(feats.cpu().numpy())
        labels.extend(batch_labels.numpy())
        sources.extend([domain_id[src] for src in batch_infos['source']])

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    sources = np.array(sources)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # reduce dimension
    if REDUCE_METHOD.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
        embeddings = reducer.fit_transform(features)
        title = "UMAP projection of ImageNet-ResNet50 features of UC dataset"
    elif REDUCE_METHOD.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
        embeddings = reducer.fit_transform(features)
        title = "t-SNE projection of ImageNet-ResNet50 features of UC dataset"
    else:
        raise ValueError("REDUCE_METHOD must be 'umap' or 'tsne'")
    
    # visualize
    unique_labels = np.unique(labels)
    markers = ['o','s','^','D']
    domain_colors = ['red', 'blue']

    plt.figure(figsize=(10,8))
    for src_val, src_name, color in zip([0,1], ['Daping', 'Shaoyifu'], domain_colors):
        for lbl_idx, lbl in enumerate(unique_labels):
            mask = (sources == src_val) & (labels == lbl)
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], c=color, marker=markers[lbl_idx], alpha=0.6, s=10, label=f'{src_name} - Class {lbl}' if src_val == 0 else "")
    plt.title('Joint View: Color=Hospital, Marker=Class')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(unique_labels)*2], labels_legend[:len(unique_labels)*2], loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('joint_class_domain.png', dpi=150)
    plt.show()

    n_classes = len(unique_labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    for ax, lbl in zip(axes, unique_labels):
        mask = labels == lbl
        sc = ax.scatter(embeddings[mask,0], embeddings[mask, 1], c=sources[mask], cmap='coolwarm', alpha=0.6, s=10)
        ax.set_title(f'Class {lbl}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        legend1 = ax.legend(*sc.legend_elements(), title='Domain')
        ax.add_artist(legend1)
    plt.tight_layout()
    plt.savefig('per_class_umap.png', dpi=150)
    plt.show()

    # 类别中心偏移分析
    domain_names = ['Daping', 'Shaoyifu']
    centers_2d = {}
    for src_val, src_name in enumerate(domain_names):
        centers = []
        for lbl in unique_labels:
            mask = (sources == src_val) & (labels == lbl)
            centers.append(features[mask].mean(axis=0))
        centers = np.stack(centers)
        centers_2d[src_name] = reducer.transform(centers)

    plt.figure(figsize=(8,6))
    for i, lbl in enumerate(unique_labels):
        plt.plot([centers_2d['Daping'][i, 0], centers_2d['Shaoyifu'][i, 0]], [centers_2d['Daping'][i, 1], centers_2d['Shaoyifu'][i, 1]], 'k--', alpha=0.5, linewidth=1)
        plt.scatter(centers_2d['Daping'][i, 0], centers_2d['Daping'][i, 1], c='blue', marker=markers[i], s=120, edgecolors='k')
        plt.scatter(centers_2d['Shaoyifu'][i, 0], centers_2d['Shaoyifu'][i, 1], c='red', marker=markers[i], s=120, edgecolors='k')
        plt.text(centers_2d['Daping'][i, 0]+0.2, centers_2d['Daping'][i, 1]+0.2, f'C{lbl}', fontsize=9)
        plt.text(centers_2d['Shaoyifu'][i, 0]+0.2, centers_2d['Shaoyifu'][i, 1]+0.2, f'C{lbl}', fontsize=9)

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0],[0], marker='o',color='w', markerfacecolor='blue', markersize=8,label='Daping'), Line2D([0],[0], marker='o',color='w', markerfacecolor='red', markersize=8,label='Shaoyifu')]
        plt.legend(handles=legend_elements, loc='best')
        plt.title('Class Centers Shift between Daping and Shaoyifu')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig('class_centers_shift.png', dpi=150)
        plt.show()