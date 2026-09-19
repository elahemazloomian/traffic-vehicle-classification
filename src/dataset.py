import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="dataset_cleaned", batch_size=32, img_size=224):
    """
    لود کردن دیتاست‌های train و val با استفاده از ImageFolder و اعمال ترنسفرم‌های استاندارد
    """
    
    # ۱. تعریف پیش‌پردازش‌ها و افزایش داده (Data Augmentation) برای آموزش
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), # برای افزایش پویایی و یادگیری بهتر مدل
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ) # نرمال‌سازی استاندارد بر اساس مقادیر ImageNet
    ])

    # ۲. پیش‌پردازش برای اعتبارسنجی (بدون Augmentation تصادفی)
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ۳. مسیر پوشه‌های داده‌های تمیز
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    # ۴. ساخت دیتاست‌ها با ImageFolder (پوشه‌ها به عنوان کلاس‌ها شناخته می‌شوند)
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transforms)

    # ۵. ساخت DataLoaderها
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    print(f"Classes detected: {train_dataset.classes}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")

    return train_loader, val_loader

if __name__ == "__main__":
    # تست صحت عملکرد اسکریپت
    train_loader, val_loader = get_data_loaders()
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break