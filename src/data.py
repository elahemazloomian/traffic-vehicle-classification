from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import build_transforms


class VehicleDataset(Dataset):
    """Reads images listed in a table (reports/splits.csv) one at a time."""

    def __init__(self, df, data_dir, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.class_names = df["class"].tolist()
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.data_dir / self.paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index]


def make_datasets(cfg):
    reports = Path(cfg["paths"]["reports_dir"])
    splits = pd.read_csv(reports / "splits.csv")
    datasets = {}
    for name in ("train", "val", "test"):
        part = splits[splits["subset"] == name].reset_index(drop=True)
        transform = build_transforms(cfg, train=(name == "train"))
        datasets[name] = VehicleDataset(part, cfg["paths"]["data_dir"], transform)
    return datasets


def make_loaders(cfg):
    datasets = make_datasets(cfg)
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])

    loaders = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=is_train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            generator=generator if is_train else None,
        )
    return loaders