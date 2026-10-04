import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small baseline: a few conv blocks, global average pooling, one linear layer."""

    def __init__(self, num_classes, channels=(32, 64, 128), dropout=0.0, pooling="max"):
        super().__init__()
        pool_layer = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d

        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                pool_layer(2),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


def build_model(cfg, num_classes):
    m = cfg["model"]
    return SimpleCNN(num_classes, tuple(m["channels"]), m["dropout"], m["pooling"])