
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "test", "unclean")


def scan_split(data_dir, split):
    """Visit every file in data_dir/split/<class>/ and describe it."""
    data_dir = Path(data_dir)
    rows = []
    for class_dir in sorted(p for p in (data_dir / split).iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            row = {
                "split": split,
                "class": class_dir.name,
                "path": path.relative_to(data_dir).as_posix(),
                "extension": path.suffix.lower(),
                "readable": False,
                "width": None,
                "height": None,
                "mode": None,
                "error": "",
            }
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as img:
                        img.load()
                        row.update(width=img.width, height=img.height,
                                   mode=img.mode, readable=True)
                except Exception as e:
                    row["error"] = str(e)
            else:
                row["error"] = "not an image extension"
            rows.append(row)
    return rows


def build_manifest(data_dir):
    rows = []
    for split in SPLITS:
        rows.extend(scan_split(data_dir, split))
    return pd.DataFrame(rows)


def print_summary(df):
    print("\n=== Images per class and split ===")
    print(df.groupby(["class", "split"]).size().unstack(fill_value=0))

    print("\n=== Class folders found in each split ===")
    for split in SPLITS:
        print(split, "->", sorted(df.loc[df["split"] == split, "class"].unique()))

    bad = df[~df["readable"]]
    print(f"\n=== Files that could not be read as images: {len(bad)} ===")
    if len(bad) > 0:
        print(bad[["path", "error"]].head(20).to_string(index=False))

    ok = df[df["readable"]]
    print("\n=== Image size statistics (pixels) ===")
    print(ok[["width", "height"]].describe().round(1))

    tiny = ok[(ok["width"] < 32) | (ok["height"] < 32)]
    print(f"\nImages smaller than 32 px on one side: {len(tiny)}")

    print("\n=== Color modes ===")
    print(ok["mode"].value_counts().to_string())


def main():
    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    df = build_manifest(data_dir)

    out = Path(cfg["paths"]["reports_dir"]) / "audit_files.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print_summary(df)
    print("\nSaved table to:", out)


if __name__ == "__main__":
    main()