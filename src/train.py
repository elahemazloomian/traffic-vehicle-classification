import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data import make_loaders
from src.engine import evaluate, train_one_epoch
from src.models import build_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils import get_device, load_config, set_seed


def apply_overrides(cfg, overrides):
    """Change settings from the command line, e.g. 'augmentation.enabled=false'."""
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown setting: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return cfg


def plot_history(history, out_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(history["epoch"], history["train_acc"], label="train accuracy")
    axes[1].plot(history["epoch"], history["val_acc"], label="val accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro-F1")
    axes[1].set_title("Accuracy and macro-F1")
    axes[2].plot(history["epoch"], history["train_r2"], label="train")
    axes[2].plot(history["epoch"], history["val_r2"], label="val")
    axes[2].set_title("R2 of probabilities (extra)")
    axes[3].plot(history["epoch"], history["lr"])
    axes[3].set_yscale("log")
    axes[3].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a model and keep the best checkpoint.")
    parser.add_argument("--name", required=True, help="name of this run, e.g. baseline")
    parser.add_argument("--set", nargs="*", default=[], help="setting overrides, e.g. train.lr=0.0003")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(), args.set)
    set_seed(cfg["seed"])
    device = get_device()
    tc = cfg["train"]

    reports = Path(cfg["paths"]["reports_dir"])
    checkpoints = Path(cfg["paths"]["checkpoints_dir"])
    (reports / "runs").mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    with open(reports / "class_to_idx.json", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    loaders = make_loaders(cfg)
    model = build_model(cfg, len(class_to_idx)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"run: {args.name} | device: {device} | parameters: {n_params}")
    print("changed settings:", args.set if args.set else "none")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = None
    if tc["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=tc["scheduler_factor"], patience=tc["scheduler_patience"]
        )
    elif tc["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    rows = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    checkpoint_path = checkpoints / f"{args.name}.pth"
    started = time.time()

    for epoch in range(1, tc["epochs"] + 1):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, train_r2 = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val = evaluate(model, loaders["val"], loss_fn, device)

        rows.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc, "train_r2": train_r2,
            "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "val_r2": val["r2"],
            "seconds": time.time() - epoch_start,
        })
        print(f"epoch {epoch:2d} | lr {lr:.5f} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val['loss']:.3f} acc {val['accuracy']:.3f} macro-F1 {val['macro_f1']:.3f} R2 {val['r2']:.3f} | "
              f"{rows[-1]['seconds']:.0f}s")

        if tc["scheduler"] == "plateau":
            scheduler.step(val["macro_f1"])
        elif scheduler is not None:
            scheduler.step()

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
        else:
            bad_epochs += 1

        if improved or tc["save"] == "last":
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "config": cfg,
                "image_size": cfg["data"]["image_size"],
                "normalize_mean": IMAGENET_MEAN,
                "normalize_std": IMAGENET_STD,
                "seed": cfg["seed"],
                "best_epoch": epoch,
                "val_macro_f1": val["macro_f1"],
                "val_accuracy": val["accuracy"],
            }, checkpoint_path)

        if not improved and tc["early_stopping_patience"] and bad_epochs >= tc["early_stopping_patience"]:
            print(f"No improvement for {bad_epochs} epochs: stopping early.")
            break

    history = pd.DataFrame(rows)
    history.to_csv(reports / "runs" / f"{args.name}_history.csv", index=False)
    plot_history(history, reports / "figures" / f"{args.name}_curves.png", args.name)

    best = history.loc[history["val_macro_f1"].idxmax()]
    summary = {
        "name": args.name,
        "changed_settings": args.set,
        "parameters": n_params,
        "epochs_run": len(history),
        "best_epoch": int(best["epoch"]),
        "val_macro_f1": round(float(best["val_macro_f1"]), 4),
        "val_accuracy": round(float(best["val_acc"]), 4),
        "val_loss": round(float(best["val_loss"]), 4),
        "val_r2": round(float(best["val_r2"]), 4),
        "train_accuracy_at_best": round(float(best["train_acc"]), 4),
        "minutes": round((time.time() - started) / 60, 1),
    }
    with open(reports / "runs" / f"{args.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBest epoch:", summary["best_epoch"], "| val macro-F1:", summary["val_macro_f1"],
          "| val accuracy:", summary["val_accuracy"])
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()