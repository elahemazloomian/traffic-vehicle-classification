import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data import make_datasets
from src.models import build_model
from src.utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Per-class scores, confusion matrix and the worst mistakes of a saved run.")
    parser.add_argument("--run", required=True, help="name of the run, e.g. a_base")
    # The test set is reserved for the final evaluation, so only validation is allowed here.
    parser.add_argument("--split", default="val", choices=["val"])
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    reports = Path(cfg["paths"]["reports_dir"])
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path(cfg["paths"]["checkpoints_dir"]) / f"{args.run}.pth",
                      weights_only=True, map_location="cpu")
    cfg["model"] = ckpt["config"]["model"]  # build exactly the model that was trained
    cfg["data"]["image_size"] = ckpt["image_size"]
    names = [n for n, _ in sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])]
    n = len(names)

    model = build_model(cfg, n).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = make_datasets(cfg)[args.split]
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    probs, y_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
            y_true += labels.tolist()
    probs = torch.cat(probs).numpy()
    y_true = np.array(y_true)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    print(f"run: {args.run} | split: {args.split} | images: {len(y_true)} | saved epoch: {ckpt['best_epoch']}\n")
    labels = list(range(n))
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(reports / "runs" / f"{args.run}_per_class_{args.split}.csv")

    # Confusion matrix: every row adds up to 1, so small and large classes can be compared.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(labels)
    ax.set_yticklabels(names)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"{args.run}: confusion matrix on {args.split} (each row adds up to 1)")
    plt.tight_layout()
    plt.savefig(figures / f"{args.run}_confusion_{args.split}.png", dpi=100)
    plt.close()

    pairs = sorted(((cm[i, j], names[i], names[j]) for i in labels for j in labels if i != j and cm[i, j] > 0),
                   reverse=True)
    print("Most frequent mistakes (true -> predicted):")
    for count, true_name, pred_name in pairs[:6]:
        print(f"  {true_name} -> {pred_name}: {count}")

    # The 12 mistakes the model was most sure about.
    wrong = np.where(y_true != y_pred)[0]
    worst = wrong[np.argsort(-conf[wrong])][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), worst):
        image = Image.open(Path(cfg["paths"]["data_dir"]) / dataset.paths[idx]).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"true: {names[y_true[idx]]}\npredicted: {names[y_pred[idx]]} ({conf[idx]:.2f})", fontsize=10)
    fig.suptitle(f"{args.run}: the {len(worst)} most confident mistakes on {args.split} "
                 f"({len(wrong)} mistakes in total)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(figures / f"{args.run}_errors_{args.split}.png", dpi=90)
    plt.close()
    print(f"\nSaved pictures and table for {args.run} in {figures} and {reports / 'runs'}")


if __name__ == "__main__":
    main()