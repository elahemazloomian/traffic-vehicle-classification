import os
from PIL import Image
import imagehash
from collections import Counter


def audit_dataset(base_dir="..", hamming_threshold=5):
    splits = ["train", "test", "unclean"]
    all_images = []
    class_counts = {split: Counter() for split in splits}
    unreadable_files = []
    unusual_sizes = []

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} not found. Please ensure the dataset is extracted correctly.")
            continue
            
        print(f"Scanning split: {split}...")
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(root, file)
                    
        
                    rel_path = os.path.relpath(img_path, split_dir)
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1:
                        class_name = path_parts[0]
                        class_counts[split][class_name] += 1

                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size

                            if width < 32 or height < 32:
                                unusual_sizes.append((img_path, (width, height)))

                            img_hash = imagehash.phash(img)
                            full_rel_path = os.path.relpath(img_path, base_dir)
                            all_images.append((full_rel_path, img_hash))
                            
                    except Exception as e:
                        unreadable_files.append((img_path, str(e)))

    
    duplicates = []
    visited = set()
    
    for i in range(len(all_images)):
        if i in visited:
            continue
        group = [all_images[i][0]]
        for j in range(i + 1, len(all_images)):
            if j in visited:
                continue
           
            hamming_dist = all_images[i][1] - all_images[j][1]
            if hamming_dist <= hamming_threshold:
                group.append(all_images[j][0])
                visited.add(j)
        if len(group) > 1:
            duplicates.append(group)


    print("\n" + "="*60)
    print("--- DATASET AUDIT REPORT (WITH HAMMING DISTANCE) ---")
    print("="*60)
    
    print(f"\n1. Unreadable or corrupted files: {len(unreadable_files)}")
    for path, err in unreadable_files:
        print(f"   - {path}: {err}")

    print(f"\n2. Unusually small images (< 32x32): {len(unusual_sizes)}")
    for path, size in unusual_sizes[:5]:
        print(f"   - {path} (Size: {size})")

    print("\n3. Class distribution per split:")
    for split, counts in class_counts.items():
        print(f"   [{split}]:")
        for cls, cnt in counts.items():
            print(f"     - {cls}: {cnt}")

    print(f"\n4. Similar/Duplicate groups found (Threshold <= {hamming_threshold}): {len(duplicates)}")
    for idx, group in enumerate(duplicates[:10]):
        print(f"   Group {idx + 1}:")
        for p in group:
            print(f"     * {p}")
    if len(duplicates) > 10:
        print(f"   ... and {len(duplicates) - 10} more groups.")

    print("\nDataset audit with Hamming distance completed successfully!")

if __name__ == "__main__":
    audit_dataset(base_dir="..")