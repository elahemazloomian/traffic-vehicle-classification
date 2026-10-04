from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import load_config, set_seed


def combine(path_a, path_b, data_dir, height=224):
    """Put two images side by side in one picture."""
    images = []
    for p in (path_a, path_b):
        img = Image.open(Path(data_dir) / p).convert("RGB")
        width = int(img.width * height / img.height)
        images.append(img.resize((width, height)))
    canvas = Image.new("RGB", (images[0].width + images[1].width + 10, height), "white")
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (images[0].width + 10, 0))
    return canvas


def plot_examples(pairs, data_dir, out_path, low, high, only_conflicts=False, n=6, seed=42):
    sub = pairs[(pairs["distance"] >= low) & (pairs["distance"] <= high)]
    if only_conflicts:
        sub = sub[~sub["same_class"]]
    if len(sub) == 0:
        print("No pairs for", out_path.name)
        return
    sub = sub.sample(min(n, len(sub)), random_state=seed)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), sub.iterrows()):
        ax.imshow(combine(row["path_a"], row["path_b"], data_dir))
        ax.set_title(
            f"distance {row['distance']} | {row['split_a']}/{row['class_a']}  vs  "
            f"{row['split_b']}/{row['class_b']}",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print("Saved", out_path)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    data_dir = cfg["paths"]["data_dir"]
    reports = Path(cfg["paths"]["reports_dir"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    max_d = cfg["duplicates"]["max_distance"]
    pairs = pd.read_csv(reports / "duplicate_pairs.csv")

    plot_examples(pairs, data_dir, figures / "dup_distance_0.png", 0, 0, seed=cfg["seed"])
    plot_examples(pairs, data_dir, figures / "dup_distance_1_4.png", 1, 4, seed=cfg["seed"])
    plot_examples(pairs, data_dir, figures / "dup_distance_5_8.png", 5, max_d, seed=cfg["seed"])
    plot_examples(pairs, data_dir, figures / "dup_label_conflicts.png", 0, max_d,
                  only_conflicts=True, seed=cfg["seed"])


if __name__ == "__main__":
    main()