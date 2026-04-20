import torchvision.transforms as T

def get_train_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),           # 肠镜图像上下翻转有意义
        T.RandomRotation(degrees=20),          # 旋转角度加大
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
        T.RandomPerspective(distortion_scale=0.3, p=0.5), # 透视变换
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])