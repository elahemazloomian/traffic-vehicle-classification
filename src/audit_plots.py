from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import load_config, set_seed


def plot_class_counts(df, out_path):
    counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(11, 5))
    ax.set_title("Number of images per class and split")
    ax.set_xlabel("class")
    ax.set_ylabel("images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_size_distribution(df, out_path):
    ok = df[df["readable"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ok["width"], bins=40)
    axes[0].set_title("Image width (pixels)")
    axes[1].hist(ok["height"], bins=40)
    axes[1].set_title("Image height (pixels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_samples(df, data_dir, out_path, n=12, seed=42):
    ok = df[(df["split"] == "train") & df["readable"]]
    sample = ok.sample(n, random_state=seed)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        img = Image.open(Path(data_dir) / row["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{row['class']}  ({row['width']}x{row['height']})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    data_dir = cfg["paths"]["data_dir"]
    reports = Path(cfg["paths"]["reports_dir"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(reports / "audit_files.csv")

    plot_class_counts(df, figures / "audit_class_counts.png")
    plot_size_distribution(df, figures / "audit_image_sizes.png")
    plot_samples(df, data_dir, figures / "audit_samples.png", seed=cfg["seed"])
    print("Saved 3 figures in:", figures)


if __name__ == "__main__":
    main()