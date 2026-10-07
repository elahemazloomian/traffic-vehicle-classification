import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.predict import load_checkpoint, predict
from src.split import build_groups
from src.utils import get_device, load_config

UNKNOWN = "unknown"


def split_pool(pool, pairs, group_distance, seed):
    """Split the Nissan pool in two halves, keeping near-duplicate images together."""
    group_ids = build_groups(pool["path"].tolist(), pairs, group_distance)
    sizes = pd.Series(group_ids).value_counts()
    rng = np.random.default_rng(seed)
    val_groups, count = set(), 0
    for group in rng.permutation(np.unique(group_ids)):
        if count >= len(pool) / 2:
            break
        val_groups.add(group)
        count += sizes[group]
    return np.where(np.isin(group_ids, list(val_groups)), "val_unknown", "test_unknown")


def outcome(pred_df, threshold):
    """The class if the model is confident enough, otherwise 'unknown'."""
    return np.where(pred_df["confidence"] >= threshold, pred_df["prediction"], UNKNOWN)


def main():
    parser = argparse.ArgumentParser(description="Choose a confidence threshold using Nissan (unknown) images.")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config()
    reports = Path(cfg["paths"]["reports_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = sorted(ckpt["class_to_idx"], key=ckpt["class_to_idx"].get)

    manifest = pd.read_csv(reports / "manifest.csv")
    pairs = pd.read_csv(reports / "duplicate_pairs.csv")
    splits = pd.read_csv(reports / "splits.csv")

    pool = manifest[manifest["status"] == "unseen_class"].reset_index(drop=True)
    pool["part"] = split_pool(pool, pairs, cfg["split"]["group_distance"], cfg["seed"])
    pool[["path", "class", "part"]].to_csv(reports / "unknown_split.csv", index=False)

    def run(paths):
        return predict(model, ckpt, [data_dir / p for p in paths], device, 0.0)

    known_val = splits[splits["subset"] == "val"].reset_index(drop=True)
    unk_val = pool[pool["part"] == "val_unknown"].reset_index(drop=True)
    unk_test = pool[pool["part"] == "test_unknown"].reset_index(drop=True)
    pred_known = run(known_val["path"])
    pred_unk_val = run(unk_val["path"])
    pred_unk_test = run(unk_test["path"])
    print(f"known val: {len(known_val)} | Nissan val-unknown: {len(unk_val)} | Nissan test-unknown: {len(unk_test)}")

    # Choose the threshold on validation data only (known val images + val-unknown Nissans).
    y_true = list(known_val["class"]) + [UNKNOWN] * len(unk_val)
    best_t, best_f1 = 0.0, -1.0
    grid = list(np.arange(0.30, 0.99, 0.01)) + [0.99, 0.995, 0.998, 0.999, 0.9995]
    for t in grid:
        y_pred = list(outcome(pred_known, t)) + list(outcome(pred_unk_val, t))
        score = f1_score(y_true, y_pred, labels=classes + [UNKNOWN], average="macro", zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(round(t, 4)), float(score)
    print(f"\nChosen threshold: {best_t} (macro-F1 over 9 outcomes on val: {best_f1:.3f})")

    def report(title, pred_df, true_classes=None):
        kept = pred_df["confidence"] >= best_t
        print(f"\n--- {title} ({len(pred_df)} images) ---")
        if true_classes is None:
            confident_vanet = ((pred_df["prediction"] == "vanet") & kept).mean()
            print(f"  called unknown (correct): {(~kept).mean():.1%}")
            print(f"  called vanet WITH confidence >= {best_t}: {confident_vanet:.1%}")
            print(f"  called vanet without any threshold: {(pred_df['prediction'] == 'vanet').mean():.1%}")
            print("  accepted predictions:", pred_df.loc[kept, "prediction"].value_counts().to_dict())
        else:
            correct = (pred_df["prediction"].values == np.asarray(true_classes)) & kept.values
            print(f"  sent to needs_review: {(~kept).mean():.1%}")
            print(f"  accuracy (rejected counted as wrong): {correct.mean():.3f}")
            print(f"  accuracy of the accepted ones: {correct.sum() / max(kept.sum(), 1):.3f}")

    report("known classes, val", pred_known, known_val["class"])
    report("Nissan, val-unknown (used to choose the threshold)", pred_unk_val)
    report("Nissan, test-unknown (not used for choosing)", pred_unk_test)

    with open(reports / "unknown_threshold.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "threshold": best_t,
            "criterion": "maximise macro-F1 over 8 classes + unknown on val (known val + val-unknown Nissans)",
            "val_macro_f1_9way": round(best_f1, 4),
            "n_known_val": len(known_val),
            "n_val_unknown": len(unk_val),
            "n_test_unknown": len(unk_test),
        }, f, indent=2)
    print("\nSaved: reports/unknown_threshold.json and reports/unknown_split.csv")


if __name__ == "__main__":
    main()