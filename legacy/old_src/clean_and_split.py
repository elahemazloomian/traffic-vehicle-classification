import os
import shutil
from PIL import Image
import imagehash
from sklearn.model_selection import train_test_split
from collections import defaultdict

def clean_and_split_dataset(base_dir=".", output_dir="dataset_cleaned", hamming_threshold=5, test_size=0.2, random_seed=42):
    splits = ["train", "test", "unclean"]
    all_records = []
    
    print("1. Scanning and collecting image records...")
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(img_path, base_dir)
                    path_parts = rel_path.split(os.sep)
                    
                    if len(path_parts) > 1:
                        class_name = path_parts[1] # نام کلاس
                        
                        # کلاس نیسان را برای آموزش/اعتبارسنجی وارد نکنیم (طبق دستورالعمل)
                        if class_name == "neysan":
                            continue
                            
                        try:
                            with Image.open(img_path) as img:
                                img_hash = imagehash.phash(img)
                                all_records.append({
                                    "path": img_path,
                                    "class": class_name,
                                    "hash": img_hash,
                                    "split": split
                                })
                        except Exception as e:
                            print(f"Skipping unreadable file {img_path}: {e}")

    print(f"Total valid images collected (excluding neysan): {len(all_records)}")

    print("2. Removing duplicates using Hamming distance...")
    drop_indices = set()
    for i in range(len(all_records)):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(all_records)):
            if j in drop_indices:
                continue
            # اگر تصاویر بسیار شبیه یا تکراری بودند، یکی را حذف می‌کنیم
            if all_records[i]["hash"] - all_records[j]["hash"] <= hamming_threshold:
                # اولویت نگهداری با train است، بعد test، بعد unclean
                drop_indices.add(j)

    clean_records = [rec for idx, rec in enumerate(all_records) if idx not in drop_indices]
    print(f"Images remaining after duplicate removal: {len(clean_records)}")

    # تفکیک بر اساس کلاس و برچسب برای تقسیم Stratified
    paths = [rec["path"] for rec in clean_records]
    labels = [rec["class"] for rec in clean_records]

    print("3. Creating 80/20 stratified train/validation split...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=test_size, random_state=random_seed, stratify=labels
    )

    # تابع کمکی برای کپی کردن فایل‌ها به ساختار جدید
    def save_to_split(file_paths, target_split_name):
        for path in file_paths:
            # استخراج نام کلاس و نام فایل
            parts = path.split(os.sep)
            class_name = parts[-2]
            file_name = parts[-1]
            
            dest_dir = os.path.join(output_dir, target_split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(path, os.path.join(dest_dir, file_name))

    # ذخیره در پوشه جدید
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    save_to_split(train_paths, "train")
    save_to_split(val_paths, "val")

    print(f"\nDataset successfully cleaned and split into '{output_dir}/train' and '{output_dir}/val'!")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

if __name__ == "__main__":
    clean_and_split_dataset()