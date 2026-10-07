import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.predict import load_checkpoint, predict
from src.unknown_eval import UNKNOWN, outcome
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description="Final evaluation on the test set. Run it ONCE.")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config()
    reports = Path(cfg["paths"]["reports_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = sorted(ckpt["class_to_idx"], key=ckpt["class_to_idx"].get)
    threshold = json.load(open(reports / "unknown_threshold.json", encoding="utf-8"))["threshold"]

    splits = pd.read_csv(reports / "splits.csv")
    test = splits[splits["subset"] == "test"].reset_index(drop=True)
    pool = pd.read_csv(reports / "unknown_split.csv")
    nissan = pool[pool["part"] == "test_unknown"].reset_index(drop=True)

    def run(paths):
        return predict(model, ckpt, [data_dir / p for p in paths], device, 0.0)

    pred_known = run(test["path"])
    pred_nissan = run(nissan["path"])
    labels9 = classes + [UNKNOWN]

    # (1) the 8 known classes only, plain argmax
    acc1 = accuracy_score(test["class"], pred_known["prediction"])
    f1_1 = f1_score(test["class"], pred_known["prediction"], labels=classes, average="macro", zero_division=0)
    print(f"=== (1) test, 8 classes only: {len(test)} images ===")
    print(f"accuracy {acc1:.4f} | macro-F1 {f1_1:.4f}")
    print(classification_report(test["class"], pred_known["prediction"], labels=classes, digits=3, zero_division=0))

    # (2) test + Nissans, whose correct answer is 'unknown'
    y_true = list(test["class"]) + [UNKNOWN] * len(nissan)
    y_plain = list(pred_known["prediction"]) + list(pred_nissan["prediction"])
    y_thr = list(outcome(pred_known, threshold)) + list(outcome(pred_nissan, threshold))
    acc_plain = accuracy_score(y_true, y_plain)
    acc_thr = accuracy_score(y_true, y_thr)
    f1_thr = f1_score(y_true, y_thr, labels=labels9, average="macro", zero_division=0)
    acc_known_thr = accuracy_score(test["class"], outcome(pred_known, threshold))

    print(f"=== (2) test + {len(nissan)} Nissan images (correct answer: unknown) ===")
    print(f"no threshold (Nissans always wrong): accuracy {acc_plain:.4f}")
    print(f"threshold {threshold}: accuracy {acc_thr:.4f} | macro-F1 over 9 outcomes {f1_thr:.4f}")
    print(f"known test images only, with threshold (rejected = wrong): accuracy {acc_known_thr:.4f}")
    print(f"Nissans correctly rejected: {(pred_nissan['confidence'] < threshold).mean():.1%}")

    with open(reports / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint, "threshold": threshold,
            "n_test_known": len(test), "n_test_nissan": len(nissan),
            "known_only_accuracy": round(acc1, 4), "known_only_macro_f1": round(f1_1, 4),
            "with_nissan_no_threshold_accuracy": round(acc_plain, 4),
            "with_nissan_threshold_accuracy": round(acc_thr, 4),
            "with_nissan_threshold_macro_f1_9way": round(f1_thr, 4),
        }, f, indent=2)
    print("\nSaved: reports/final_test_results.json")


if __name__ == "__main__":
    main()