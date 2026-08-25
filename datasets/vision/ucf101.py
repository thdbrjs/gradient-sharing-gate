import json
import os
import re

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, UCF101
from torchvision.transforms.functional import to_pil_image
from .utils import DatasetSpec, split_dataset


dataset_dir = "ucf101"
template = "a photo of a person doing {}."


class UCF101FrameDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.classes = [
            re.sub(r"(?<!^)(?=[A-Z])", " ", name) for name in dataset.classes
        ]
        self.targets = []
        for index in range(len(dataset)):
            video_index, _ = dataset.video_clips.get_clip_location(index)
            sample_index = dataset.indices[video_index]
            self.targets.append(dataset.samples[sample_index][1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video, _, label = self.dataset[index]
        frame = to_pil_image(video[0].permute(2, 0, 1))
        return self.transform(frame), label


class UCF101ExtractedSplit(Dataset):
    def __init__(self, image_root, records, classes, transform):
        self.image_root = image_root
        self.records = records
        self.classes = classes
        self.targets = [record[1] for record in records]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative_path, label, _ = self.records[index]
        path = os.path.join(self.image_root, *relative_path.split("/"))
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def _clean_class_names(dataset):
    dataset.classes = [
        re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ")
        for name in dataset.classes
    ]
    return dataset


def _build_extracted_frames(dataset_root, train_transform, test_transform):
    """Load pre-extracted first frames without decoding the UCF videos."""
    split_path = os.path.join(dataset_root, "split_zhou_UCF101.json")
    midframes_root = os.path.join(dataset_root, "UCF-101-midframes")
    if os.path.isfile(split_path) and os.path.isdir(midframes_root):
        with open(split_path, encoding="utf-8") as file:
            splits = json.load(file)
        all_records = splits["train"] + splits["val"] + splits["test"]
        names_by_label = {record[1]: record[2] for record in all_records}
        classes = [names_by_label[label] for label in sorted(names_by_label)]
        return (
            UCF101ExtractedSplit(midframes_root, splits["train"], classes, train_transform),
            UCF101ExtractedSplit(midframes_root, splits["val"], classes, test_transform),
            UCF101ExtractedSplit(midframes_root, splits["test"], classes, test_transform),
        )

    frames_root = os.path.join(dataset_root, "frames")
    train_root = os.path.join(frames_root, "train")
    test_root = os.path.join(frames_root, "test")

    if os.path.isdir(train_root):
        train_dataset = _clean_class_names(ImageFolder(train_root, transform=train_transform))
        eval_dataset = _clean_class_names(ImageFolder(train_root, transform=test_transform))
        if os.path.isdir(test_root):
            test_dataset = _clean_class_names(ImageFolder(test_root, transform=test_transform))
            if test_dataset.class_to_idx != train_dataset.class_to_idx:
                raise ValueError("UCF101 frame train/test class folders must match")
            return split_dataset(train_dataset, eval_dataset, test_dataset)
        return split_dataset(train_dataset, eval_dataset)

    # A flat frames/<class>/*.jpg tree is also accepted and split 70/10/20.
    if os.path.isdir(frames_root):
        train_dataset = _clean_class_names(ImageFolder(frames_root, transform=train_transform))
        eval_dataset = _clean_class_names(ImageFolder(frames_root, transform=test_transform))
        return split_dataset(train_dataset, eval_dataset)
    raise FileNotFoundError(f"Extracted UCF101 frames not found under {frames_root}")


def build_ucf101(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    video_root = os.path.join(dataset_root, "UCF-101")
    annotation_root = os.path.join(dataset_root, "ucfTrainTestlist")

    if not (os.path.isdir(video_root) and os.path.isdir(annotation_root)):
        return _build_extracted_frames(dataset_root, train_transform, test_transform)

    # Dataset 파일 수동 다운로드 필요: https://www.crcv.ucf.edu/data/UCF101.php
    train_dataset = UCF101(
        video_root,
        annotation_root,
        frames_per_clip=1,
        step_between_clips=10**9,
        fold=1,
        train=True,
    )
    test_dataset = UCF101(
        video_root,
        annotation_root,
        frames_per_clip=1,
        step_between_clips=10**9,
        fold=1,
        train=False,
    )
    return split_dataset(
        UCF101FrameDataset(train_dataset, train_transform),
        UCF101FrameDataset(train_dataset, test_transform),
        UCF101FrameDataset(test_dataset, test_transform),
    )


DATASET = DatasetSpec(build_ucf101, template)
