import json
from pathlib import Path

import torch
import torch.nn as nn

from src.data import make_loaders
from src.models import build_model
from src.utils import get_device, load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()

    with open(Path(cfg["paths"]["reports_dir"]) / "class_to_idx.json", encoding="utf-8") as f:
        num_classes = len(json.load(f))

    model = build_model(cfg, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("device:", device, "| classes:", num_classes, "| trainable parameters:", n_params)

    images, labels = next(iter(make_loaders(cfg)["train"]))
    images, labels = images.to(device), labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Check 1: the shapes are right and the starting loss is near ln(8) = 2.08.
    model.eval()
    with torch.no_grad():
        scores = model(images)
    print("\nOutput shape:", tuple(scores.shape), "(expected: batch size x classes)")
    print(f"Loss before training: {loss_fn(scores, labels).item():.3f} (random guessing gives about 2.08)")

    # Check 2: the model must be able to memorize ONE batch. If it cannot, the code has a bug.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nTraining on the same batch of 32 images:")
    for step in range(1, 101):
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()