import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UCDataset(Dataset):
    """
    to load uc data from UC_all_divide_data.xlsx

    Args:
        excel_path (str): path to the excel file
        base_dir (str): base directory of the images
        transform (callable, optional): preprocessing transform to be applied on the image
        split (str or None): split to load, 'train' or 'val'
        source (str or list or None): select the source to load, like '大坪UC', ['大坪UC', '邵逸夫UC']， None denotes all sources
        label_filter (int or list or None): select specific labels to load, like 0, [0, 1], None denotes all labels
        return_info (bool): whether to return the info of the image, like (patient_ID, source, path), default False just return (image, label)
    """

    def __init__(self, excel_path, base_dir, transform=None, split=None, source=None, label_filter=None, return_info=False):
        self.base_dir = base_dir
        self.transform = transform
        self.return_info = return_info

        df = pd.read_excel(excel_path)

        if split is not None:
            df = df[df['remark'] == split].reset_index(drop=True)

        if source is not None:
            if isinstance(source, str):
                source = [source]
            df = df[df['image_source'].isin(source)].reset_index(drop=True)

        if label_filter is not None:
            if isinstance(label_filter, (int, float)):
                label_filter = [label_filter]
            df = df[df['label'].isin(label_filter)].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = row['label']

        if self.return_info:
            info = {
                'patient_id': row['patient_ID'],
                'source': row['image_source'],
                'path': row['image_path']
            }
            return image, label, info
        else:
            return image, label