import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_of(path):
    """Fingerprint of the raw file bytes: the same picture always gives the same text."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


SPLITS = ("train", "test", "unclean")


def scan(folder):
    """Return {relative path: md5} for every image inside the train, test and unclean folders."""
    folder = Path(folder)
    found = {}
    for split in SPLITS:
        if not (folder / split).is_dir():
            raise SystemExit(f"Folder not found: {folder / split}")
        for p in sorted((folder / split).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                found[p.relative_to(folder).as_posix()] = md5_of(p)
    return found

def compare(old, new):
    """List every image that is no longer at its old place: moved somewhere else, or deleted."""
    new_by_md5 = defaultdict(list)
    for path, md5 in new.items():
        new_by_md5[md5].append(path)

    rows = []
    for path, md5 in old.items():
        if new.get(path) == md5:
            continue
        moved_to = [p for p in new_by_md5.get(md5, []) if old.get(p) != md5]
        rows.append({
            "old_path": path,
            "md5": md5,
            "change": "moved" if moved_to else "deleted",
            "new_path": moved_to[0] if moved_to else "",
        })
    return pd.DataFrame(rows, columns=["old_path", "md5", "change", "new_path"])


def main():
    parser = argparse.ArgumentParser(description="Compare the data folder with a backup and record what changed.")
    parser.add_argument("--backup", required=True, help="path of the untouched backup copy of the data folder")
    args = parser.parse_args()

    cfg = load_config()
    reports = Path(cfg["paths"]["reports_dir"])
    old = scan(args.backup)
    new = scan(cfg["paths"]["data_dir"])
    print(f"images in backup: {len(old)} | images now: {len(new)}")

    changes = compare(old, new)
    if changes.empty:
        print("No changes found.")
    else:
        where_from = changes["old_path"].str.split("/").str[:2].str.join("/")
        where_to = changes["new_path"].apply(lambda p: "/".join(p.split("/")[:2]) if p else "(deleted)")
        print("\n=== What changed (from -> to) ===")
        print(changes.groupby([where_from.rename("from"), where_to.rename("to")]).size().to_string())

    old_md5 = set(old.values())
    added = [p for p, m in new.items() if m not in old_md5]
    if added:
        print(f"\nWARNING: {len(added)} image(s) exist now that were not in the backup, e.g. {added[:3]}")

    out = reports / "data_changes.csv"
    changes.to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()