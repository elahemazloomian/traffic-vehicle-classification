import torch.nn as nn
from torchvision import models as tv


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


class ResNet18Classifier(nn.Module):
    """ImageNet-pretrained ResNet18 with a new 8-class head.

    feature_extraction: only the new head (fc) is trained.
    finetune: layer4 and the new head are trained, everything else stays frozen.
    Frozen parts are kept in eval mode, so their BatchNorm statistics never change.
    """

    def __init__(self, num_classes, mode="finetune", dropout=0.0):
        super().__init__()
        if mode not in ("feature_extraction", "finetune"):
            raise ValueError(f"unknown mode: {mode}")
        net = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))

        self.trainable_parts = ["fc"] if mode == "feature_extraction" else ["layer4", "fc"]
        for name, param in net.named_parameters():
            param.requires_grad = name.split(".")[0] in self.trainable_parts
        self.net = net

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, module in self.net.named_children():
                if name not in self.trainable_parts:
                    module.eval()
        return self

    def forward(self, x):
        return self.net(x)


def build_model(cfg, num_classes):
    m = cfg["model"]
    if m["name"] == "resnet18":
        return ResNet18Classifier(num_classes, m["mode"], m["dropout"])
    return SimpleCNN(num_classes, tuple(m["channels"]), m["dropout"], m["pooling"])