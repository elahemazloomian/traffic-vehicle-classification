import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import load_config

COLS, ROWS = 4, 5  # 20 images per sheet
REVIEWABLE = ["keep", "unclean_candidate", "unseen_class", "test_frozen"]


def main():
    parser = argparse.ArgumentParser(description="Make contact sheets so images can be checked by eye.")
    parser.add_argument("--split", required=True, choices=["train", "test", "unclean"])
    parser.add_argument("--cls", required=True, help="class folder name, e.g. vanet")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    reports = Path(cfg["paths"]["reports_dir"])

    hashes = pd.read_csv(reports / "hashes.csv")
    manifest = pd.read_csv(reports / "manifest.csv")[["path", "status"]]
    table = hashes.merge(manifest, on="path")

    part = table[(table["split"] == args.split) & (table["class"] == args.cls)
                 & table["status"].isin(REVIEWABLE)].reset_index(drop=True)
    if part.empty:
        raise SystemExit(f"No images found for split={args.split}, class={args.cls}")

    out_dir = reports / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    n_sheets = math.ceil(len(part) / per_sheet)

    for sheet in range(n_sheets):
        chunk = part.iloc[sheet * per_sheet:(sheet + 1) * per_sheet]
        fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.2, ROWS * 3.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), chunk.iterrows()):
            ax.imshow(Image.open(data_dir / row["path"]).convert("RGB"))
            ax.set_title(Path(row["path"]).stem, fontsize=13, fontweight="bold")
        fig.suptitle(f"{args.split} / {args.cls}   sheet {sheet + 1} of {n_sheets}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out_dir / f"{args.split}_{args.cls}_{sheet + 1:02d}.png", dpi=80)
        plt.close()

    print(f"{len(part)} images -> {n_sheets} sheets in {out_dir}")


if __name__ == "__main__":
    main()