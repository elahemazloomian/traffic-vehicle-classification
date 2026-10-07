from pathlib import Path

import pandas as pd

from src.utils import load_config

GROUPS = {
    "random": ["i_random", "i_random_seed43", "i_random_seed44"],
    "balanced": ["i_balanced", "i_balanced_seed43", "i_balanced_seed44"],
}


def main():
    runs_dir = Path(load_config()["paths"]["reports_dir"]) / "runs"
    means = {}
    for label, names in GROUPS.items():
        recalls = []
        for name in names:
            df = pd.read_csv(runs_dir / f"{name}_per_class_val.csv")
            class_col = "class" if "class" in df.columns else df.columns[0]
            recall_col = next(c for c in df.columns if "recall" in c.lower())
            recalls.append(df.set_index(class_col)[recall_col])
        means[label] = pd.concat(recalls, axis=1).mean(axis=1)
    table = pd.DataFrame(means)
    table["balanced - random"] = table["balanced"] - table["random"]
    print(table.round(3))


if __name__ == "__main__":
    main()