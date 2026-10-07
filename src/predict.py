import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.models import build_model
from src.transforms import build_transforms
from src.utils import get_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(target):
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["config"], len(ckpt["class_to_idx"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, ckpt, paths, device, threshold, batch_size=32):
    idx_to_class = {i: name for name, i in ckpt["class_to_idx"].items()}
    # Same preprocessing as in evaluation, rebuilt from the config stored in the checkpoint.
    transform = build_transforms(ckpt["config"], train=False)
    rows = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = []
        for path in chunk:
            with Image.open(path) as img:
                batch.append(transform(img.convert("RGB")))
        probs = torch.softmax(model(torch.stack(batch).to(device)), dim=1).cpu()
        top2 = probs.topk(2, dim=1)
        for path, values, indices in zip(chunk, top2.values, top2.indices):
            confidence = float(values[0])
            rows.append({
                "path": str(path),
                "prediction": idx_to_class[int(indices[0])],
                "confidence": round(confidence, 4),
                "second_choice": idx_to_class[int(indices[1])],
                "second_confidence": round(float(values[1]), 4),
                "decision": "accept" if confidence >= threshold else "needs_review",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Classify vehicle images with a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="e.g. checkpoints/r_ft.pth")
    parser.add_argument("--images", required=True, help="an image file or a folder of images")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="confidence below this value is marked needs_review (0 = never)")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    table = predict(model, ckpt, paths, device, args.threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(table.head(10).to_string(index=False))
    print(f"\n{len(table)} images | decisions: {table['decision'].value_counts().to_dict()}")
    print("predicted classes:", table["prediction"].value_counts().to_dict())
    print("Saved:", args.output)


if __name__ == "__main__":
    main()