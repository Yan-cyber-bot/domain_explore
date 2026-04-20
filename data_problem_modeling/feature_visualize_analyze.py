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
REDUCE_METHOD = 'tsne'
RANDOM_SEED = 42

dataset = UCDataset(excel_path=os.path.join(DATA_DIR, 'UC_all_divided_data.xlsx'), base_dir=DATA_DIR, transform=get_val_transforms(IMG_SIZE),
                    source=['大坪UC', '邵逸夫UC'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(DEVICE)
model.eval()

domain_id = {'大坪UC': 'Daping', '邵逸夫UC': 'Shaoyifu'}
features = []
labels = []

print("Extracting features...")
with torch.no_grad():
    for images, batch_labels in tqdm(dataloader):
        images = images.to(DEVICE)
        feats = model(images).squeeze(-1).squeeze(-1)
        features.append(feats.cpu().numpy())
        labels.extend(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
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
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=10)

    domain_names = list(domain_id.values())
    handles = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=scatter.cmap(scatter.norm(i)),markersize=8)
               for i in range(len(domain_names))]
    plt.legend(handles, domain_names, title='Domain')

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("domain_shift_visualization_tsne.png", dpi=150)
    plt.show()

    # Calculate the center of each domain
    centers = []
    for i in range(len(domain_names)):
        center = np.mean(embeddings[labels == i], axis=0)
        centers.append(center)
        print(f"Center of {domain_names[i]}: ({center[0]:.4f}, {center[1]:.4f})")

    if len(centers) == 2:
        dist = np.linalg.norm(centers[0] - centers[1])
        print(f"Distance between centers: {dist:.4f}")
        
