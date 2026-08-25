import json
import os

from torchvision.datasets import ImageFolder, ImageNet
from .utils import DatasetSpec, split_dataset


dataset_dir = "imagenet"
template = "a photo of a {}."


def _display_names(dataset_root, folders):
    """Return prompt-friendly names for a (possibly partial) ImageNet tree."""
    mapping = {}
    json_path = os.path.join(dataset_root, "classnames.json")
    text_path = os.path.join(dataset_root, "classnames.txt")
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as file:
            mapping = json.load(file)
    elif os.path.isfile(text_path):
        with open(text_path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, separator, value = line.partition("\t")
                mapping[key] = value if separator else key
    return [mapping.get(folder, folder.replace("_", " ").replace("-", " ")) for folder in folders]


def _build_imagefolder_subset(dataset_root, train_transform, test_transform):
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(
            f"ImageNet subset not found: expected class folders under {train_root}"
        )

    train_dataset = ImageFolder(train_root, transform=train_transform)
    eval_dataset = ImageFolder(train_root, transform=test_transform)
    display_names = _display_names(dataset_root, train_dataset.classes)
    train_dataset.classes = display_names
    eval_dataset.classes = display_names

    if os.path.isdir(val_root):
        test_dataset = ImageFolder(val_root, transform=test_transform)
        if test_dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError("ImageNet subset train/val class folders must match")
        test_dataset.classes = display_names
        return split_dataset(train_dataset, eval_dataset, test_dataset)
    return split_dataset(train_dataset, eval_dataset)


def build_imagenet(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    # A plain ImageFolder tree is enough for experiments on a partial ImageNet.
    # Prefer it when no torchvision ImageNet metadata archive is present.
    if not os.path.isfile(os.path.join(dataset_root, "meta.bin")):
        return _build_imagefolder_subset(dataset_root, train_transform, test_transform)

    train_dataset = ImageNet(dataset_root, split="train", transform=train_transform)
    val_dataset = ImageNet(dataset_root, split="train", transform=test_transform)
    test_dataset = ImageNet(dataset_root, split="val", transform=test_transform)
    train_dataset.classes = [names[0] for names in train_dataset.classes]
    val_dataset.classes = train_dataset.classes
    test_dataset.classes = train_dataset.classes
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_imagenet, template)
