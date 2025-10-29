import os
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

CONFIG_PATH = "./config.json"

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "./input/car_data/car_data"
    train_dir: str = os.path.join(data_dir, "train")
    test_dir: str = os.path.join(data_dir, "test")
    img_size: int = 400
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 10
    model_dir: str = "./models"
    model_name: str = "car_classifier_resnet34.pth"

    # preprocessing-related defaults
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    horizontal_flip: bool = True
    rotation_deg: int = 15
    pin_memory: bool = True

def _patch_from_json(cfg: Config, json_path: str) -> Config:
    """
    If json_path exists, load and override fields in cfg.
    Only keys present in the JSON will be patched.
    """
    if not os.path.isfile(json_path):
        return cfg

    with open(json_path, "r") as f:
        data = json.load(f)

    # convert list->tuple for mean/std if provided
    if "normalize_mean" in data and isinstance(data["normalize_mean"], list):
        data["normalize_mean"] = tuple(data["normalize_mean"])
    if "normalize_std" in data and isinstance(data["normalize_std"], list):
        data["normalize_std"] = tuple(data["normalize_std"])

    # patch only existing fields on the dataclass
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            # ignore unknown keys or log a warning
            pass

    return cfg

# create default config and try to patch from JSON automatically at import time
cfg = Config()
try:
    cfg = _patch_from_json(cfg, CONFIG_PATH)
except Exception as e:
    # keep default config on failure (don't crash import)
    print(f"Warning: failed to load config.json: {e}")

# ----------------- transforms and dataset utilities -----------------

def resize_transform(img_size: int) -> transforms.Resize:
    """Resize (and optionally center-crop later if you want)."""
    return transforms.Resize((img_size, img_size))

def augmentation_transforms(horizontal_flip: bool = True, rotation_deg: int = 15) -> transforms.Compose:
    """Return augmentation transforms (applied only for training)."""
    aug_list = []
    if horizontal_flip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if rotation_deg and rotation_deg > 0:
        aug_list.append(transforms.RandomRotation(rotation_deg))
    # add augmentations (color jitter, cutout, random erasing, etc.)
    return transforms.Compose(aug_list)

def tensor_and_normalize(mean: Tuple[float,float,float], std: Tuple[float,float,float]) -> transforms.Compose:
    """Convert to tensor and normalize."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def build_train_transforms(cfg: Config) -> transforms.Compose:
    """Compose transforms for training dataset."""
    parts = [
        resize_transform(cfg.img_size),
        augmentation_transforms(cfg.horizontal_flip, cfg.rotation_deg),
        tensor_and_normalize(cfg.normalize_mean, cfg.normalize_std)
    ]
    # flatten any nested Compose produced by augmentation_transforms
    flattened = []
    for p in parts:
        if isinstance(p, transforms.Compose):
            flattened.extend(p.transforms)
        else:
            flattened.append(p)
    return transforms.Compose(flattened)

def build_test_transforms(cfg: Config) -> transforms.Compose:
    """Compose transforms for validation/test dataset (no augmentation)."""
    return transforms.Compose([
        resize_transform(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalize_mean, cfg.normalize_std)
    ])

def build_datasets(cfg: Config,
                   train_subdir: str = "train",
                   test_subdir: str = "test",
                   train_transform: Optional[transforms.Compose] = None,
                   test_transform: Optional[transforms.Compose] = None
                   ) -> Dict[str, datasets.ImageFolder]:
    """Create ImageFolder datasets using provided transforms."""
    train_dir = os.path.join(cfg.data_dir, train_subdir)
    test_dir = os.path.join(cfg.data_dir, test_subdir)

    if train_transform is None:
        train_transform = build_train_transforms(cfg)
    if test_transform is None:
        test_transform = build_test_transforms(cfg)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    return {"train": train_dataset, "test": test_dataset}

def build_dataloaders(datasets_dict: Dict[str, datasets.ImageFolder],
                      cfg: Config,
                      shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """Create DataLoader objects from datasets."""
    train_loader = DataLoader(
        datasets_dict["train"],
        batch_size=cfg.batch_size,
        shuffle=shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        datasets_dict["test"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    return {"train": train_loader, "test": test_loader}

# convenience function to load a config from a different path at runtime
def load_config(path: str) -> Config:
    c = Config()
    return _patch_from_json(c, path)

# test usage:
if __name__ == "__main__":
    print("Active config:")
    print(asdict(cfg))

    train_tf = build_train_transforms(cfg)
    test_tf  = build_test_transforms(cfg)

    datasets_dict = build_datasets(cfg)
    loaders = build_dataloaders(datasets_dict, cfg)

    print("Train dataset size:", len(datasets_dict["train"]))
    print("Test dataset size:", len(datasets_dict["test"]))
    print("Train transform:", train_tf)
    print("Test transform:", test_tf)
