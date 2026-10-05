import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import load_config, set_seed


def build_groups(paths, pairs, max_distance):
    """Connect images whose fingerprints are close and return one group id per image."""
    index = {p: i for i, p in enumerate(paths)}
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    close = pairs[pairs["distance"] <= max_distance]
    for a, b in zip(close["path_a"], close["path_b"]):
        if a in index and b in index:
            root_a, root_b = find(index[a]), find(index[b])
            if root_a != root_b:
                parent[root_b] = root_a

    roots = [find(i) for i in range(len(paths))]
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def make_split(labels, group_ids, val_fraction, seed):
    """Pick about val_fraction of the images as validation, keeping each group on one side."""
    n_splits = round(1 / val_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(labels))
    _, val_idx = next(splitter.split(dummy, labels, groups=group_ids))
    subset = np.array(["train"] * len(labels), dtype=object)
    subset[val_idx] = "val"
    return subset


def run_checks(splits, pairs, hashes, group_distance, n_test=None):
    n_classes = splits["class"].nunique()

    # 1. A group must never be on both sides.
    tv = splits[splits["subset"].isin(["train", "val"])]
    sides = tv.groupby("group")["subset"].nunique()
    assert (sides == 1).all(), "a group is split between train and val"

    # 2. Test is untouched and shares no path with train/val.
    test_paths = set(splits.loc[splits["subset"] == "test", "path"])

    assert n_test is None or len(test_paths) == n_test, "test size changed"
    assert test_paths.isdisjoint(set(tv["path"])), "test image found in train/val"

    # 3. Every class appears in every subset.
    for name, part in splits.groupby("subset"):
        assert part["class"].nunique() == n_classes, f"class missing in {name}"

    # 4. No close pair (<= group_distance) crosses the three subsets.
    where = dict(zip(splits["path"], splits["subset"]))
    close = pairs[pairs["distance"] <= group_distance]
    crossing = [
        (a, b) for a, b in zip(close["path_a"], close["path_b"])
        if a in where and b in where and where[a] != where[b]
    ]
    assert len(crossing) == 0, f"{len(crossing)} close pairs cross subsets"

    # 5. No byte-identical file appears in two subsets.
    merged = splits.merge(hashes[["path", "md5"]], on="path")
    assert (merged.groupby("md5")["subset"].nunique() == 1).all(), "identical file in two subsets"

    # Information only: weaker look-alikes that remain across train and val.
    weak = pairs[(pairs["distance"] > group_distance)]
    weak_cross = sum(
        1 for a, b in zip(weak["path_a"], weak["path_b"])
        if {where.get(a), where.get(b)} == {"train", "val"}
    )
    print(f"\nAll checks passed. (Info: {weak_cross} weaker look-alike pairs "
          f"with distance > {group_distance} still cross train/val; most are different vehicles.)")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    reports = Path(cfg["paths"]["reports_dir"])
    group_distance = cfg["split"]["group_distance"]
    assert group_distance <= cfg["duplicates"]["max_distance"]

    manifest = pd.read_csv(reports / "manifest.csv")
    pairs = pd.read_csv(reports / "duplicate_pairs.csv")
    hashes = pd.read_csv(reports / "hashes.csv")

    keep = manifest[manifest["status"] == "keep"].reset_index(drop=True)
    test = manifest[manifest["status"] == "test_frozen"].reset_index(drop=True)

    classes = sorted(keep["class"].unique())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    assert set(test["class"]) == set(classes), "test and train classes differ"

    keep["label"] = keep["class"].map(class_to_idx)
    keep["group"] = build_groups(keep["path"].tolist(), pairs, group_distance)
    keep["subset"] = make_split(keep["label"].values, keep["group"].values,
                                cfg["data"]["val_fraction"], cfg["seed"])

    test["label"] = test["class"].map(class_to_idx)
    test["group"] = -1
    test["subset"] = "test"

    columns = ["path", "class", "label", "subset", "group"]
    splits = pd.concat([keep[columns], test[columns]], ignore_index=True)
    splits.to_csv(reports / "splits.csv", index=False)
    with open(reports / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print("=== Images per subset ===")
    print(splits["subset"].value_counts().to_string())

    print("\n=== Images per class and subset ===")
    table = splits.groupby(["class", "subset"]).size().unstack(fill_value=0)
    table["val %"] = (100 * table["val"] / (table["train"] + table["val"])).round(1)
    print(table)

    sizes = keep.groupby("group").size()
    print(f"\n=== Groups (close images, distance <= {group_distance}) ===")
    print("groups:", len(sizes), "| groups with more than one image:", int((sizes > 1).sum()),
          "| largest group:", int(sizes.max()), "images")

    run_checks(splits, pairs, hashes, group_distance, len(test))
    print("\nSaved:", reports / "splits.csv", "and", reports / "class_to_idx.json")


if __name__ == "__main__":
    main()