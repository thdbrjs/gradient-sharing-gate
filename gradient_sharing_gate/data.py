"""Base-to-new 16-shot pipelines for the focused CLIP experiments."""

from math import ceil
from random import Random

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import DTD, EuroSAT, FGVCAircraft
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


GLOBAL_SEED = 2026
TEMPLATES = {
    "fgvc": "a photo of a {}, a type of aircraft.",
    "eurosat": "a centered satellite photo of {}.",
    "dtd": "{} texture.",
}
EUROSAT_NAMES = {
    "AnnualCrop": "Annual Crop Land",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "SeaLake": "Sea or Lake",
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClassSubset(Dataset):
    def __init__(self, dataset, classes):
        labels = dataset_labels(dataset)
        label_map = {label: new_label for new_label, label in enumerate(classes)}
        self.dataset = dataset
        self.indices = [index for index, label in enumerate(labels) if label in label_map]
        self.targets = [label_map[labels[index]] for index in self.indices]
        self.classes = [dataset.classes[index] for index in classes]
        self.template = dataset.template

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, self.targets[index]


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        labels = dataset_labels(dataset.dataset)
        return [labels[index] for index in dataset.indices]
    for attribute in ("targets", "_labels", "y"):
        labels = getattr(dataset, attribute, None)
        if labels is not None:
            return labels
    return [label for _, label in dataset._samples]


def fewshot_subset(dataset, shots):
    indices_by_class = {label: [] for label in range(len(dataset.classes))}
    for index, label in enumerate(dataset_labels(dataset)):
        indices_by_class[label].append(index)
    rng = Random(GLOBAL_SEED)
    selected = []
    for candidates in indices_by_class.values():
        selected.extend(
            rng.sample(candidates, shots)
            if len(candidates) >= shots
            else rng.choices(candidates, k=shots)
        )
    subset = Subset(dataset, selected)
    subset.classes = dataset.classes
    subset.template = dataset.template
    return subset


def base2new_split(train, validation, test):
    split = ceil(len(train.classes) / 2)
    base, new = range(split), range(split, len(train.classes))
    return (
        ClassSubset(train, base),
        ClassSubset(validation, base),
        ClassSubset(validation, new),
        ClassSubset(test, base),
        ClassSubset(test, new),
    )


def _subset(dataset, indices):
    subset = Subset(dataset, indices)
    subset.classes = dataset.classes
    subset.template = dataset.template
    return subset


def _split_single_dataset(train, evaluation):
    generator = torch.Generator().manual_seed(GLOBAL_SEED)
    indices = torch.randperm(len(train), generator=generator).tolist()
    train_end = int(len(indices) * 0.7)
    validation_end = int(len(indices) * 0.8)
    return (
        _subset(train, indices[:train_end]),
        _subset(evaluation, indices[train_end:validation_end]),
        _subset(evaluation, indices[validation_end:]),
    )


def _transforms():
    train = Compose([
        RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])
    test = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return train, test


def _build_datasets(dataset_name, root, train_transform, test_transform):
    if dataset_name == "fgvc":
        dataset_root = f"{root}/fgvc_aircraft"
        datasets = (
            FGVCAircraft(dataset_root, split="train", annotation_level="variant", download=True, transform=train_transform),
            FGVCAircraft(dataset_root, split="val", annotation_level="variant", download=True, transform=test_transform),
            FGVCAircraft(dataset_root, split="test", annotation_level="variant", download=True, transform=test_transform),
        )
    elif dataset_name == "dtd":
        dataset_root = f"{root}/dtd"
        datasets = (
            DTD(dataset_root, split="train", download=True, transform=train_transform),
            DTD(dataset_root, split="val", download=True, transform=test_transform),
            DTD(dataset_root, split="test", download=True, transform=test_transform),
        )
    elif dataset_name == "eurosat":
        dataset_root = f"{root}/eurosat"
        train = EuroSAT(dataset_root, download=True, transform=train_transform)
        evaluation = EuroSAT(dataset_root, download=True, transform=test_transform)
        train.classes = [EUROSAT_NAMES.get(name, name) for name in train.classes]
        evaluation.classes = train.classes
        for dataset in (train, evaluation):
            dataset.template = TEMPLATES[dataset_name]
        return _split_single_dataset(train, evaluation)
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return datasets


def get_dataloader(batch_size, dataset_name="fgvc", root="data", shots=16, setting="base2new"):
    if setting != "base2new":
        raise ValueError("this repository supports only the base2new setting")
    train_transform, test_transform = _transforms()
    train, validation, test = _build_datasets(
        dataset_name, root, train_transform, test_transform
    )
    for dataset in (train, validation, test):
        dataset.template = TEMPLATES[dataset_name]
    train, val_base, val_new, test_base, test_new = base2new_split(train, validation, test)
    train = fewshot_subset(train, shots)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        (DataLoader(val_base, batch_size=batch_size), DataLoader(val_new, batch_size=batch_size)),
        (DataLoader(test_base, batch_size=batch_size), DataLoader(test_new, batch_size=batch_size)),
        len(train.classes),
    )
