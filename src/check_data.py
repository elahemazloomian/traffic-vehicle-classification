from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data import VehicleDataset, make_datasets, make_loaders
from src.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.utils import load_config, set_seed


def denormalize(x):
    """Undo the normalization so a tensor can be shown as a normal picture."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def save_augmentation_figure(cfg, out_path, n_images=4, n_versions=4):
    reports = Path(cfg["paths"]["reports_dir"])
    splits = pd.read_csv(reports / "splits.csv")
    train_df = splits[splits["subset"] == "train"].sample(n_images, random_state=cfg["seed"])
    data_dir = cfg["paths"]["data_dir"]
    plain = VehicleDataset(train_df, data_dir, build_transforms(cfg, train=False))
    augmented = VehicleDataset(train_df, data_dir, build_transforms(cfg, train=True))

    fig, axes = plt.subplots(n_images, n_versions + 1,
                             figsize=(3 * (n_versions + 1), 3 * n_images))
    for row in range(n_images):
        axes[row, 0].imshow(denormalize(plain[row][0]).permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"original ({plain.class_names[row]})", fontsize=9)
        for col in range(1, n_versions + 1):
            axes[row, col].imshow(denormalize(augmented[row][0]).permute(1, 2, 0).numpy())
            axes[row, col].set_title("augmented", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    datasets = make_datasets(cfg)
    loaders = make_loaders(cfg)

    for name, loader in loaders.items():
        print(f"{name}: {len(datasets[name])} images, {len(loader)} batches")

    images, labels = next(iter(loaders["train"]))
    size = cfg["data"]["image_size"]
    print("\nBatch shape:", tuple(images.shape), "| dtype:", images.dtype)
    print(f"Pixel mean {images.mean():.2f}, std {images.std():.2f} (should be near 0 and 1)")
    print("Labels in this batch:", sorted(labels.tolist()))

    n_classes = len(set(datasets["train"].labels))
    assert tuple(images.shape[1:]) == (3, size, size)
    assert 0 <= labels.min() and labels.max() < n_classes

    # Validation must be identical every time; training must change when augmentation is on.
    assert torch.equal(datasets["val"][0][0], datasets["val"][0][0]), "val is not stable"
    if cfg["augmentation"]["enabled"]:
        assert not torch.equal(datasets["train"][0][0], datasets["train"][0][0]), \
            "train images are not changing"
    print("\nChecks passed: val is stable, train is augmented.")

    out = Path(cfg["paths"]["reports_dir"]) / "figures" / "augmentation_examples.png"
    save_augmentation_figure(cfg, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()