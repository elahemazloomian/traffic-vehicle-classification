import argparse
import subprocess
import sys
from pathlib import Path

from src.utils import load_config

# Group 1 (old recipe): reduce the learning rate when the validation score stalls, stop early.
BASE = ["train.epochs=60", "model.channels=[32,64,128,256,256]"]

EXPERIMENTS = {
    "a_base": [],
    "a_base_seed43": ["seed=43"],
    "a_base_seed44": ["seed=44"],
    "a_dropout03": ["model.dropout=0.3"],
    "a_dropout03_seed43": ["model.dropout=0.3", "seed=43"],
    "a_dropout03_seed44": ["model.dropout=0.3", "seed=44"],
    "a_no_aug": ["augmentation.enabled=false"],
    "a_no_hflip": ["augmentation.hflip=0.0"],
    "a_dropout05": ["model.dropout=0.5"],
    "a_wd0": ["train.weight_decay=0.0"],
    "a_wd1e3": ["train.weight_decay=0.001"],
    "a_avgpool": ["model.pooling=avg"],
    "a_no_scheduler": ["train.scheduler=none"],
}

# Group 2 (fixed recipe): 40 epochs, cosine learning-rate decay, no early stopping, keep the LAST epoch.
# Nothing depends on the validation set while training, so scores are less noisy and the same
# recipe can later be used on all the data.
FIXED_BASE = [
    "model.channels=[32,64,128,256,256]",
    "train.epochs=40",
    "train.scheduler=cosine",
    "train.early_stopping_patience=0",
    "train.save=last",
]

FIXED_EXPERIMENTS = {
    "f_base": [],
    "f_base_seed43": ["seed=43"],
    "f_base_seed44": ["seed=44"],
    "f_dropout03": ["model.dropout=0.3"],
    "f_dropout03_seed43": ["model.dropout=0.3", "seed=43"],
    "f_dropout03_seed44": ["model.dropout=0.3", "seed=44"],
    "f_no_aug": ["augmentation.enabled=false"],
    "f_no_hflip": ["augmentation.hflip=0.0"],
    "f_wd0": ["train.weight_decay=0.0"],
    "f_wd1e3": ["train.weight_decay=0.001"],
    "f_avgpool": ["model.pooling=avg"],
    "f_constant_lr": ["train.scheduler=none"],
}
# Group 3 (ResNet18 transfer learning): same fixed recipe as group 2.
RESNET_COMMON = [
    "model.name=resnet18",
    "train.epochs=40",
    "train.scheduler=cosine",
    "train.early_stopping_patience=0",
    "train.save=last",
]

RESNET_EXPERIMENTS = {
    "r_fe": ["model.mode=feature_extraction"],
    "r_fe_seed43": ["model.mode=feature_extraction", "seed=43"],
    "r_fe_seed44": ["model.mode=feature_extraction", "seed=44"],
    "r_ft": ["model.mode=finetune"],
    "r_ft_seed43": ["model.mode=finetune", "seed=43"],
    "r_ft_seed44": ["model.mode=finetune", "seed=44"],
}
# Group 4 (imbalance): keep 20% of the minibus and taxi training images, compare batch samplers.
IMBALANCE = ["data.imbalance_classes=[minibus,taxi]", "data.imbalance_keep=0.2"]

IMBALANCE_EXPERIMENTS = {
    "i_random": IMBALANCE + ["train.sampler=random"],
    "i_random_seed43": IMBALANCE + ["train.sampler=random", "seed=43"],
    "i_random_seed44": IMBALANCE + ["train.sampler=random", "seed=44"],
    "i_balanced": IMBALANCE + ["train.sampler=balanced"],
    "i_balanced_seed43": IMBALANCE + ["train.sampler=balanced", "seed=43"],
    "i_balanced_seed44": IMBALANCE + ["train.sampler=balanced", "seed=44"],
}
def main():
    all_experiments = {**EXPERIMENTS, **FIXED_EXPERIMENTS, **RESNET_EXPERIMENTS, **IMBALANCE_EXPERIMENTS}

    parser = argparse.ArgumentParser(description="Run several experiments one after another.")
    parser.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    names = args.only if args.only else list(all_experiments)
    unknown = [n for n in names if n not in all_experiments]
    if unknown:
        sys.exit(f"Unknown experiment(s): {unknown}. Choose from: {list(all_experiments)}")

    runs_dir = Path(load_config()["paths"]["reports_dir"]) / "runs"
    for i, name in enumerate(names, start=1):
        if (runs_dir / f"{name}_summary.json").exists():
            print(f"[{i}/{len(names)}] skip {name}: already finished")
            continue
        if name in RESNET_EXPERIMENTS:
            base = RESNET_COMMON
        elif name in FIXED_EXPERIMENTS or name in IMBALANCE_EXPERIMENTS:
            base = FIXED_BASE
        else:
            base = BASE
        command = [sys.executable, "-m", "src.train", "--name", name, "--set", *base, *all_experiments[name]]
        print(f"\n[{i}/{len(names)}] {name}: {' '.join(command[3:])}")
        if args.dry_run:
            continue
        if subprocess.run(command).returncode != 0:
            sys.exit(f"{name} failed. Fix the problem, then run this script again: finished runs are skipped.")

    print("\nAll requested experiments are done. Next: python -m src.summarize_runs")


if __name__ == "__main__":
    main()