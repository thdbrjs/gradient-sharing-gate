import os
from torchvision.datasets import ImageFolder, SUN397
from .utils import DatasetSpec, split_dataset


dataset_dir = "sun397"
template = "a photo of a {}."


def build_sun397(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    # Compact mirrors may provide an ordinary train/test ImageFolder tree.
    imagefolder_root = os.path.join(dataset_root, "sun397")
    train_root = os.path.join(imagefolder_root, "train")
    test_root = os.path.join(imagefolder_root, "test")
    if os.path.isdir(train_root):
        train_dataset = ImageFolder(train_root, transform=train_transform)
        val_dataset = ImageFolder(train_root, transform=test_transform)
        train_dataset.classes = [name.replace("_", " ") for name in train_dataset.classes]
        val_dataset.classes = train_dataset.classes
        if os.path.isdir(test_root):
            test_dataset = ImageFolder(test_root, transform=test_transform)
            if test_dataset.class_to_idx != train_dataset.class_to_idx:
                raise ValueError("SUN397 train/test class folders must match")
            test_dataset.classes = train_dataset.classes
            return split_dataset(train_dataset, val_dataset, test_dataset)
        return split_dataset(train_dataset, val_dataset)

    train_dataset = SUN397(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    eval_test_dataset = SUN397(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, eval_test_dataset)


DATASET = DatasetSpec(build_sun397, template)
