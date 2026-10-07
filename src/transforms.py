from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg, train):
    """Steps applied to every image. Augmentation is used only for the training subset."""
    size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]

    steps = [T.Resize((size, size))]
    if train and aug["enabled"]:
        steps += [
            T.RandomHorizontalFlip(p=aug["hflip"]),
            T.RandomAffine(
                degrees=aug["rotation_degrees"],
                translate=(aug["translate"], aug["translate"]),
                scale=tuple(aug["scale"]),
            ),
            T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        ]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)