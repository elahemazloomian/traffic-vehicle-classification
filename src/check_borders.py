from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config


def bottom_brightness(path, rows=3):
    """Average brightness (0 black, 255 white) of the last few pixel rows of an image."""
    with Image.open(path) as img:
        return float(np.asarray(img.convert("L"))[-rows:].mean())


def main():
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    splits = pd.read_csv(Path(cfg["paths"]["reports_dir"]) / "splits.csv")

    splits["bright_bottom"] = [bottom_brightness(data_dir / p) > 200 for p in splits["path"]]

    print("Percent of images with a bright bottom edge, per subset:")
    print(splits.groupby("subset")["bright_bottom"].mean().mul(100).round(1).to_string())

    print("\nPercent per class and subset:")
    table = splits.groupby(["class", "subset"])["bright_bottom"].mean().unstack().mul(100).round(1)
    print(table[["train", "val", "test"]])


if __name__ == "__main__":
    main()