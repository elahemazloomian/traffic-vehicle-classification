import argparse
import subprocess
import sys
from pathlib import Path

from src.utils import load_config

# Every experiment starts from this base. Then it changes ONE thing, so any difference is caused by that thing.
BASE = ["train.epochs=60", "model.channels=[32,64,128,256,256]"]

EXPERIMENTS = {
    # Repeats of the base: they show how much the score moves by chance alone.
    "a_base": [],
    "a_base_seed43": ["seed=43"],
    "a_base_seed44": ["seed=44"],
    # One change at a time.
    "a_no_aug": ["augmentation.enabled=false"],
    "a_no_hflip": ["augmentation.hflip=0.0"],
    "a_dropout03": ["model.dropout=0.3"],
    "a_dropout05": ["model.dropout=0.5"],
    "a_wd0": ["train.weight_decay=0.0"],
    "a_wd1e3": ["train.weight_decay=0.001"],
    "a_avgpool": ["model.pooling=avg"],
    "a_no_scheduler": ["train.scheduler=none"],
}


def main():
    parser = argparse.ArgumentParser(description="Run several experiments one after another.")
    parser.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    names = args.only if args.only else list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"Unknown experiment(s): {unknown}. Choose from: {list(EXPERIMENTS)}")

    runs_dir = Path(load_config()["paths"]["reports_dir"]) / "runs"
    for i, name in enumerate(names, start=1):
        if (runs_dir / f"{name}_summary.json").exists():
            print(f"[{i}/{len(names)}] skip {name}: already finished")
            continue
        command = [sys.executable, "-m", "src.train", "--name", name, "--set", *BASE, *EXPERIMENTS[name]]
        print(f"\n[{i}/{len(names)}] {name}: {' '.join(command[3:])}")
        if args.dry_run:
            continue
        if subprocess.run(command).returncode != 0:
            sys.exit(f"{name} failed. Fix the problem, then run this script again: finished runs are skipped.")

    print("\nAll requested experiments are done. Next: python -m src.summarize_runs")


if __name__ == "__main__":
    main()