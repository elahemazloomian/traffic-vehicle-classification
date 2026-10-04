from pathlib import Path

import pandas as pd

from src.utils import load_config


def mark(manifest, paths, status):
    """Give `status` to these paths, but only if they have no status yet (first reason wins)."""
    todo = manifest["path"].isin(set(paths)) & (manifest["status"] == "")
    manifest.loc[todo, "status"] = status


def build_status_table(hashes, pairs, near_test_distance, unseen_classes):
    manifest = hashes[["split", "class", "path"]].copy()
    manifest["status"] = ""

    # Rule 1: test images are never changed.
    mark(manifest, manifest.loc[manifest["split"] == "test", "path"], "test_frozen")

    # Rule 2: unseen classes are kept aside for later analysis.
    unseen = manifest["class"].isin(unseen_classes) & (manifest["split"] != "test")
    mark(manifest, manifest.loc[unseen, "path"], "unseen_class")

    # Rule 3: anything that looks like a test image is dropped from train/unclean.
    close = pairs["distance"] <= near_test_distance
    a_is_test = pairs["split_a"] == "test"
    b_is_test = pairs["split_b"] == "test"
    near_a = pairs.loc[close & b_is_test & ~a_is_test, "path_a"]
    near_b = pairs.loc[close & a_is_test & ~b_is_test, "path_b"]
    mark(manifest, pd.concat([near_a, near_b]), "drop_near_test")

    exact = pairs[pairs["distance"] == 0]

    # Rule 4: identical-looking images with different labels -> the label cannot be trusted.
    conflict = exact[~exact["same_class"]]
    mark(manifest, pd.concat([conflict["path_a"], conflict["path_b"]]), "drop_label_conflict")

    # Rule 5: identical-looking images with the same label -> keep only the first one.
    same = exact[exact["same_class"]]
    mark(manifest, same["path_b"], "drop_duplicate")

    # Everything else is usable.
    left = manifest["status"] == ""
    manifest.loc[left & (manifest["split"] == "train"), "status"] = "keep"
    manifest.loc[left & (manifest["split"] == "unclean"), "status"] = "unclean_candidate"
    return manifest


def print_report(manifest):
    print("\n=== Status of every image ===")
    print(manifest.groupby(["split", "status"]).size().unstack(fill_value=0).T)

    kept = manifest[manifest["status"] == "keep"]
    print("\n=== Training images kept, per class ===")
    print(kept["class"].value_counts().sort_index().to_string())


def main():
    cfg = load_config()
    reports = Path(cfg["paths"]["reports_dir"])
    dup_cfg = cfg["duplicates"]
    assert dup_cfg["near_test_distance"] <= dup_cfg["max_distance"]

    hashes = pd.read_csv(reports / "hashes.csv")
    pairs = pd.read_csv(reports / "duplicate_pairs.csv")

    manifest = build_status_table(
        hashes, pairs, dup_cfg["near_test_distance"], dup_cfg["unseen_classes"]
    )
    out = reports / "manifest.csv"
    manifest.to_csv(out, index=False)

    print_report(manifest)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()