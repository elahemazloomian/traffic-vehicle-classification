import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config

SPLIT_ORDER = {"train": 0, "test": 1, "unclean": 2}


def file_md5(path):
    """Fingerprint of the raw file bytes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hashes(df, data_dir):
    """Compute an exact fingerprint (md5) and a visual fingerprint (phash) per image."""
    data_dir = Path(data_dir)
    md5s, phashes = [], []
    for i, rel_path in enumerate(df["path"], start=1):
        full_path = data_dir / rel_path
        md5s.append(file_md5(full_path))
        with Image.open(full_path) as img:
            phashes.append(str(imagehash.phash(img.convert("RGB"))))
        if i % 500 == 0:
            print(f"  hashed {i}/{len(df)} images")
    out = df[["split", "class", "path"]].copy()
    out["md5"] = md5s
    out["phash"] = phashes
    return out


def hamming_pairs(phashes, max_distance):
    """Return (i, j, distance) for every pair of images that are close enough."""
    h = np.array([int(x, 16) for x in phashes], dtype=np.uint64)
    found = []
    for i in range(len(h) - 1):
        xor = np.bitwise_xor(h[i], h[i + 1:])
        bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for k in np.where(bits <= max_distance)[0]:
            found.append((i, i + 1 + int(k), int(bits[k])))
    return found


def find_pairs(hashes, max_distance):
    found = hamming_pairs(hashes["phash"], max_distance)
    a = hashes.iloc[[i for i, _, _ in found]].reset_index(drop=True)
    b = hashes.iloc[[j for _, j, _ in found]].reset_index(drop=True)
    pairs = pd.DataFrame({
        "path_a": a["path"], "path_b": b["path"],
        "split_a": a["split"], "split_b": b["split"],
        "class_a": a["class"], "class_b": b["class"],
        "distance": [d for _, _, d in found],
        "same_md5": a["md5"] == b["md5"],
    })
    # Put the pair in a fixed order (train, test, unclean) so tables are easy to read.
    swap = pairs["split_a"].map(SPLIT_ORDER) > pairs["split_b"].map(SPLIT_ORDER)
    for col in ["path", "split", "class"]:
        pairs.loc[swap, [col + "_a", col + "_b"]] = pairs.loc[swap, [col + "_b", col + "_a"]].values
    pairs["same_class"] = pairs["class_a"] == pairs["class_b"]
    return pairs


def print_summary(hashes, pairs, max_distance):
    groups = hashes.groupby("md5").filter(lambda g: len(g) > 1)
    print("\n=== Exact duplicates (identical files) ===")
    print("groups:", groups["md5"].nunique(), "| images inside those groups:", len(groups))

    for limit in sorted({0, 4, max_distance}):
        sub = pairs[pairs["distance"] <= limit]
        print(f"\n=== Pairs with distance <= {limit}: {len(sub)} pairs ===")
        if len(sub) == 0:
            continue
        table = sub.groupby(["split_a", "split_b", "same_class"]).size().unstack(fill_value=0)
        table = table.rename(columns={True: "same label", False: "LABEL CONFLICT"})
        print(table)


def main():
    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    reports = Path(cfg["paths"]["reports_dir"])
    max_distance = cfg["duplicates"]["max_distance"]

    hashes_path = reports / "hashes.csv"
    if hashes_path.exists():
        hashes = pd.read_csv(hashes_path)
        print("Loaded saved hashes from", hashes_path)
    else:
        files = pd.read_csv(reports / "audit_files.csv")
        files = files[files["readable"]].reset_index(drop=True)
        print("Computing hashes for", len(files), "images (about a minute)...")
        hashes = compute_hashes(files, data_dir)
        hashes.to_csv(hashes_path, index=False)

    pairs = find_pairs(hashes, max_distance)
    pairs.to_csv(reports / "duplicate_pairs.csv", index=False)

    print_summary(hashes, pairs, max_distance)
    print("\nSaved:", hashes_path, "and", reports / "duplicate_pairs.csv")


if __name__ == "__main__":
    main()