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

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = '../data'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
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
patient_ids = []

print("Extracting features...")
with torch.no_grad():
    for images, _, batch_infos in tqdm(dataloader):
        images = images.to(DEVICE)
        feats = model(images).squeeze(-1).squeeze(-1)
        features.append(feats.cpu().numpy())
        sources.extend([domain_id[src] for src in batch_infos['source']])
        patient_ids.extend([pid for pid in batch_infos['patient_id']])

    features = np.concatenate(features, axis=0)
    sources = np.array(sources)
    patient_ids = np.array(patient_ids)
    print(f"Features shape: {features.shape}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    auc_scores = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(8,6))
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(features_scaled, sources, groups=patient_ids)):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = sources[train_idx], sources[test_idx]
        
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)

        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_val)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"\n5-fold CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    if mean_auc > 0.85:
        verdict = "Domain shift 极其显著"
    elif mean_auc > 0.8:
        verdict = "Domain shift 明显"
    elif mean_auc > 0.7:
        verdict = "Domain shift 中等"
    else:
        verdict = "Domain shift 较弱"

    print(f"\n结论：AUC = {mean_auc:.3f}, {verdict}")
