import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .utils import DatasetSpec


TEMPLATES = {
    "eurosat": "a centered satellite photo of {}.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "dtd": "{} texture.",
}


class CompetitionTrainDataset(Dataset):
    """대회의 labeled Base training CSV를 읽는 Dataset."""

    def __init__(
        self,
        root,
        dataset_name,
        transform=None,
        csv_name="train_16shot.csv",
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.transform = transform

        classes_path = self.root / "classes.csv"
        train_path = self.root / csv_name

        if not classes_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {classes_path}")

        if not train_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {train_path}")

        # classes.csv에 기재된 순서대로 Base 클래스를 정의
        with classes_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as file:
            class_rows = [
                row
                for row in csv.DictReader(file)
                if row["dataset"] == dataset_name
                and row["class_split"] == "base"
            ]

        if not class_rows:
            raise RuntimeError(
                f"classes.csv에서 {dataset_name} Base 클래스를 찾지 못했습니다."
            )

        self.class_keys = [row["class_key"] for row in class_rows]
        self.classes = [row["class_name"] for row in class_rows]

        self.class_to_index = {
            class_key: index
            for index, class_key in enumerate(self.class_keys)
        }

        # 선택한 데이터셋의 labeled Base 이미지만 읽음
        with train_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as file:
            self.samples = [
                row
                for row in csv.DictReader(file)
                if row["dataset"] == dataset_name
                and row["class_key"] in self.class_to_index
            ]

        self._samples = [
            (row["image_path"], self.class_to_index[row["class_key"]])
            for row in self.samples
        ]

        if not self.samples:
            raise RuntimeError(
                f"{csv_name}에서 dataset={dataset_name}인 학습 데이터를 "
                "찾지 못했습니다."
            )

        print(
            f"[Competition] dataset={dataset_name}, "
            f"classes={len(self.classes)}, "
            f"samples={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image_path = self.root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_index[row["class_key"]]
        return image, label


def _build_competition_dataset(
    root,
    dataset_name,
    train_transform,
    test_transform,
):
    # 학습용 transform
    train_dataset = CompetitionTrainDataset(
        root=root,
        dataset_name=dataset_name,
        transform=train_transform,
    )

    # 평가용 transform
    eval_dataset = CompetitionTrainDataset(
        root=root,
        dataset_name=dataset_name,
        transform=test_transform,
    )

    # 현재는 대회 train 데이터를 평가에도 사용한다.
    # 따라서 출력 정확도는 일반화 성능이 아니라 동작 확인용이다.
    return train_dataset, eval_dataset, eval_dataset


def build_competition_eurosat(
    root,
    train_transform,
    test_transform,
):
    return _build_competition_dataset(
        root,
        "eurosat",
        train_transform,
        test_transform,
    )


def build_competition_fgvc(
    root,
    train_transform,
    test_transform,
):
    return _build_competition_dataset(
        root,
        "fgvc_aircraft",
        train_transform,
        test_transform,
    )


def build_competition_dtd(
    root,
    train_transform,
    test_transform,
):
    return _build_competition_dataset(
        root,
        "dtd",
        train_transform,
        test_transform,
    )


COMPETITION_EUROSAT = DatasetSpec(
    build_competition_eurosat,
    TEMPLATES["eurosat"],
)

COMPETITION_FGVC = DatasetSpec(
    build_competition_fgvc,
    TEMPLATES["fgvc_aircraft"],
)

COMPETITION_DTD = DatasetSpec(
    build_competition_dtd,
    TEMPLATES["dtd"],
)