import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
import sys
sys.path.append('/root/autodl-tmp/domain_explore/')
from src.datasets import UCDataset, get_train_transforms, get_val_transforms

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
IMG_SIZE = 224

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Loading datasets...")

train_dataset = UCDataset(
    excel_path=os.path.join(DATA_DIR, 'UC_all_divided_data2.xlsx'),
    base_dir=DATA_DIR,
    transform=get_train_transforms(IMG_SIZE),
    split='train',
    source='邵逸夫UC',
    return_info=True
)

shaoyifu_test_dataset = UCDataset(
    excel_path=os.path.join(DATA_DIR, 'UC_all_divided_data2.xlsx'),
    base_dir=DATA_DIR,
    transform=get_val_transforms(IMG_SIZE),
    split='test',
    source='邵逸夫UC',
    return_info=True
)

daping_test_dataset = UCDataset(
    excel_path=os.path.join(DATA_DIR, 'UC_all_divided_data2.xlsx'),
    base_dir=DATA_DIR,
    transform=get_val_transforms(IMG_SIZE),
    split=None,
    source='大坪UC',
    return_info=True
)
print(f"Training samples (Shaoyifu train): {len(train_dataset)}")
print(f"Test samples (Shaoyifu test): {len(daping_test_dataset)}")
print(f"Test samples (Daping all): {len(shaoyifu_test_dataset)}")

train_patient_ids = np.array([train_dataset.df.iloc[i]['patient_ID'] for i in range(len(train_dataset))])
train_labels = np.array([train_dataset.df.iloc[i]['label'] for i in range(len(train_dataset))])

from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
sign = 0
for train_idx, val_idx in sgkf.split(np.zeros(len(train_dataset)), train_labels, groups=train_patient_ids):
    break
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)

print(f"Training subset after patient-wise split: {len(train_subset)}")
print(f"Validation subset after patient-wise split: {len(val_subset)}")

from torch.utils.data import WeightedRandomSampler
train_labels_for_sampler = train_labels[train_idx]
class_sample_counts = np.bincount(train_labels_for_sampler)
weights = 1.0 / class_sample_counts[train_labels_for_sampler]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
daping_test_loader = DataLoader(daping_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
shaoyifu_test_loader = DataLoader(shaoyifu_test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

num_classes = len(train_dataset.df['label'].unique())
print(f"Number of classes: {num_classes}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(DEVICE)

for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'bn1' in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

def train_one_epoch(loader):
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in  loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

best_val_loss =  float('inf')
patience_counter = 0
best_model_state = None

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(train_loader)
    val_loss, val_acc, _, _ = evaluate(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} |" f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 7:
            print("Early stopping triggered")
            break

model.load_state_dict(best_model_state)
print("Training completed.")

def detailed_evaluate(loader, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=f'Evaluating {dataset_name}'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_macro = np.nan
        auc_weighted = np.nan
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    return {
        'dataset':dataset_name,
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'confusion_matrix': cm,
        'classification_report': report
    }

print("\n" + "="*50)
print("Evaluating on Shaoyifu Test Set (In-Domain)")
print("="*50)
Shaoyifu_results = detailed_evaluate(shaoyifu_test_loader, "Daping Test")

print("\n" + "="*50)
print("Evaluating on Daping All Data (Cross-Domain)")
print("="*50)
Daping_results = detailed_evaluate(daping_test_loader, "Shaoyifu All")

def print_results_summary(res1, res2):
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'auc_macro', 'auc_weighted']
    print('\n' + "="*70)
    print("Cross-Domain Performance Summary")
    print("="*70)
    print(f"{'Metric':<20} | {'Shaoyifu (In-Domain)':<18} | {'Daping (Cross)':<18} | {'Drop (%)':<10}")
    print("-"*70)

    summary = {}
    for m in metrics:
        val1 = res1[m]
        val2 = res2[m]
        if not np.isnan(val1) and not np. isnan(val2):
            drop = (val1-val2)/val1 * 100
        else:
            drop = np.nan
        print(f"{m:<20} | {val1:<18.4f} | {val2:<18.4f} | {drop:<10.2f}")
        summary[m] = {'in_domain': val1, 'cross_domain': val2, 'drop_percent': drop}

    print("\nConfusion Matrix (Shaoyifu Test):")
    print(res1['confusion_matrix'])
    print("\nConfusion Matrix (Daping All):")
    print(res2['confusion_matrix'])

    print("\nClassification Report (Shaoyifu Test):")
    print(res1['classification_report'])
    print("\nClassification Report (Daping All):")
    print(res2['classification_report'])

    return summary

summary =  print_results_summary(Shaoyifu_results, Daping_results)

import json
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

serializable_summary = {k: {kk: convert_to_serializable(vv) for kk, vv in v.items()} for k, v in summary.items()}
with open('cross_domain_summary.json', 'w') as f:
    json.dump(serializable_summary, f, indent=4)

print("\nResults saved to cross_domain_summary.json")