import math
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


def plot_pair_grid(rows, data_dir, out_path, title, cols=3, max_pairs=12, seed=42):
    """Draw pairs side by side. If there are more than max_pairs, pick a random sample."""
    total = len(rows)
    if total == 0:
        print("No pairs for", out_path.name)
        return
    if total > max_pairs:
        rows = rows.sample(max_pairs, random_state=seed)
    n_rows = math.ceil(len(rows) / cols)

    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 6, n_rows * 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(combine(row["path_a"], row["path_b"], data_dir))
        ax.set_title(
            f"d={row['distance']} | {row['split_a']}/{row['class_a']}  vs  "
            f"{row['split_b']}/{row['class_b']}",
            fontsize=9,
        )
    fig.suptitle(f"{title} (showing {len(rows)} of {total} pairs)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=90)
    plt.close()
    print("Saved", out_path)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    seed = cfg["seed"]
    data_dir = cfg["paths"]["data_dir"]
    reports = Path(cfg["paths"]["reports_dir"])
    out = reports / "figures" / "explore"
    out.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(reports / "duplicate_pairs.csv")
    print("Number of pairs at each exact distance:")
    print(pairs["distance"].value_counts().sort_index().to_string())

    # 1. Single small distances: where do pairs stop being the same picture?
    for d in (1, 2, 3):
        plot_pair_grid(pairs[pairs["distance"] == d], data_dir,
                       out / f"pairs_distance_{d}.png", f"Distance {d}", seed=seed)

    # 2. Every train-vs-test pair with distance <= 4 (possible test leakage)
    cross = pairs[(pairs["split_a"] == "train") & (pairs["split_b"] == "test")
                  & (pairs["distance"] <= 4)]
    plot_pair_grid(cross, data_dir, out / "train_vs_test_le4.png",
                   "train vs test, distance <= 4", max_pairs=30, seed=seed)

    # 3. Every pair that looks alike but has different labels, distance <= 4
    conflicts = pairs[(~pairs["same_class"]) & (pairs["distance"] <= 4)]
    plot_pair_grid(conflicts, data_dir, out / "label_conflicts_le4.png",
                   "different labels, distance <= 4", max_pairs=30, seed=seed)


if __name__ == "__main__":
    main()