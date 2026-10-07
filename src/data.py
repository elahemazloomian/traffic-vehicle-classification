from pathlib import Path
import math
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

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

class BalancedBatchSampler(Sampler):
    """Every batch holds the same number of images from each class (small classes are repeated)."""

    def __init__(self, labels, batch_size, seed):
        self.by_class = {}
        for index, label in enumerate(labels):
            self.by_class.setdefault(label, []).append(index)
        self.per_class = batch_size // len(self.by_class)
        self.n_batches = math.ceil(len(labels) / batch_size)  # same epoch length as random batches
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for indices in self.by_class.values():
                picks = torch.randint(len(indices), (self.per_class,), generator=self.generator)
                batch += [indices[i] for i in picks.tolist()]
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[i] for i in order]

    def __len__(self):
        return self.n_batches


def simulate_imbalance(df, cfg):
    """Keep only a fraction of the training images of the chosen classes (same subset for every seed)."""
    classes, keep = cfg["data"]["imbalance_classes"], cfg["data"]["imbalance_keep"]
    if not classes or keep >= 1.0:
        return df
    parts = []
    for name, group in df.groupby("class"):
        if name in classes:
            group = group.sample(frac=keep, random_state=0)
        parts.append(group)
    return pd.concat(parts).reset_index(drop=True)

def make_datasets(cfg):
    reports = Path(cfg["paths"]["reports_dir"])
    splits = pd.read_csv(reports / "splits.csv")
    datasets = {}
    for name in ("train", "val", "test"):
        part = splits[splits["subset"] == name].reset_index(drop=True)
        if name == "train":
            part = simulate_imbalance(part, cfg)        
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
        if is_train and cfg["train"]["sampler"] == "balanced":
            sampler = BalancedBatchSampler(dataset.labels, cfg["data"]["batch_size"], cfg["seed"])
            loaders[name] = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=torch.cuda.is_available(),
            )
            continue
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=is_train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            generator=generator if is_train else None,
        )
    return loaders