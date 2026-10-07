import json
from pathlib import Path

import pandas as pd

from src.utils import load_config

SKIP_PREFIXES = ("quick", "smoke")


def summarize_run(summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = pd.read_csv(summary_path.with_name(summary["name"] + "_history.csv"))
    last = history.tail(5)  # the last 5 epochs: less sensitive to one lucky epoch than the best one

    train_acc_last = last["train_acc"].mean()
    val_acc_last = last["val_acc"].mean()
    return {
        "run": summary["name"],
        "changes": " ".join(summary["changed_settings"]) or "-",
        "params": summary["parameters"],
        "epochs": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
        "val_f1_best": summary["val_macro_f1"],
        "val_f1_last5": round(last["val_macro_f1"].mean(), 4),
        "val_acc_last5": round(val_acc_last, 4),
        "val_r2_best": summary.get("val_r2"),
        "train_minus_val_acc": round(train_acc_last - val_acc_last, 4),
        "minutes": summary["minutes"],
    }


def to_markdown(df):
    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def main():
    cfg = load_config()
    reports = Path(cfg["paths"]["reports_dir"])

    rows = []
    for path in sorted((reports / "runs").glob("*_summary.json")):
        if path.name.startswith(SKIP_PREFIXES):
            continue
        rows.append(summarize_run(path))

    table = pd.DataFrame(rows).sort_values("val_f1_last5", ascending=False).reset_index(drop=True)
    table.to_csv(reports / "runs_table.csv", index=False)
    (reports / "runs_table.md").write_text(to_markdown(table) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 60)
    print(table.to_string())
    print("\nSaved:", reports / "runs_table.csv", "and", reports / "runs_table.md")


if __name__ == "__main__":
    main()