"""YOLO detection annotation storage and dataset import helpers.

The GUI deliberately stores boxes as image-edge coordinates while the public
API reads and writes the normalized YOLO ``class cx cy width height`` format.
Operators therefore never need to calculate centres or sizes by hand.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import yaml

from .training import IMAGE_SUFFIXES, _class_names, _label_for_image


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("类别编号不能为负数")
        values = (self.x_center, self.y_center, self.width, self.height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("YOLO 坐标必须位于 0 到 1 之间")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("标注框宽度和高度必须大于 0")
        if self.x_center - self.width / 2 < -1e-6 or self.x_center + self.width / 2 > 1 + 1e-6:
            raise ValueError("标注框超出图像水平边界")
        if self.y_center - self.height / 2 < -1e-6 or self.y_center + self.height / 2 > 1 + 1e-6:
            raise ValueError("标注框超出图像垂直边界")

    @classmethod
    def from_corners(
        cls, class_id: int, x1: float, y1: float, x2: float, y2: float
    ) -> "YoloBox":
        left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        return cls(
            class_id=class_id,
            x_center=(left + right) / 2.0,
            y_center=(top + bottom) / 2.0,
            width=right - left,
            height=bottom - top,
        )

    def corners(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2.0,
            self.y_center - self.height / 2.0,
            self.x_center + self.width / 2.0,
            self.y_center + self.height / 2.0,
        )

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def initialize_detection_dataset(
    dataset_yaml: str | Path, class_names: Sequence[str]
) -> "YoloAnnotationProject":
    yaml_path = Path(dataset_yaml).expanduser().resolve()
    if yaml_path.exists():
        return YoloAnnotationProject(yaml_path)
    names = [str(name).strip() for name in class_names if str(name).strip()]
    if not names:
        raise ValueError("至少需要一个物品类别")
    if len(set(names)) != len(names):
        raise ValueError("物品类别名称不能重复")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (yaml_path.parent / relative).mkdir(parents=True, exist_ok=True)
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(names)},
    }
    yaml_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return YoloAnnotationProject(yaml_path)


class YoloAnnotationProject:
    """Read/write one conventional YOLO detection dataset."""

    def __init__(self, dataset_yaml: str | Path) -> None:
        self.yaml_path = Path(dataset_yaml).expanduser().resolve()
        if not self.yaml_path.is_file():
            raise ValueError(f"数据集 YAML 不存在：{self.yaml_path}")
        self._reload()

    def _reload(self) -> None:
        self.raw = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        self.class_names: Dict[int, str] = _class_names(self.raw.get("names"))
        root_value = Path(str(self.raw.get("path", "."))).expanduser()
        self.root = root_value.resolve() if root_value.is_absolute() else (
            self.yaml_path.parent / root_value
        ).resolve()
        self.image_dirs = {
            split: self._single_directory(self.raw.get(split), split) for split in ("train", "val")
        }
        for image_dir in self.image_dirs.values():
            image_dir.mkdir(parents=True, exist_ok=True)
            self._label_dir(image_dir).mkdir(parents=True, exist_ok=True)

    def _single_directory(self, value: object, split: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.yaml 的 {split} 必须是一个图片目录")
        path = Path(value).expanduser()
        path = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        if path.suffix.lower() == ".txt":
            raise ValueError("可视化标注工具要求 train/val 指向目录，不能使用图片清单 TXT")
        return path

    @staticmethod
    def _label_dir(image_dir: Path) -> Path:
        parts = list(image_dir.parts)
        indexes = [index for index, value in enumerate(parts) if value == "images"]
        if indexes:
            parts[indexes[-1]] = "labels"
            return Path(*parts)
        return image_dir.parent / f"{image_dir.name}_labels"

    def images(self, split: str) -> List[Path]:
        self._check_split(split)
        return sorted(
            path for path in self.image_dirs[split].rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )

    def read_boxes(self, image: str | Path) -> List[YoloBox]:
        image_path = Path(image).resolve()
        label = _label_for_image(image_path)
        if not label.exists():
            return []
        boxes: List[YoloBox] = []
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number} 不是YOLO检测框格式")
            try:
                box = YoloBox(int(fields[0]), *(float(value) for value in fields[1:]))
            except ValueError as exc:
                raise ValueError(f"{label}:{line_number} 标签错误：{exc}") from exc
            if box.class_id not in self.class_names:
                raise ValueError(f"{label}:{line_number} 的类别 {box.class_id} 不存在")
            boxes.append(box)
        return boxes

    def save_boxes(self, image: str | Path, boxes: Iterable[YoloBox]) -> Path:
        image_path = Path(image).resolve()
        if not image_path.is_file():
            raise ValueError(f"图片不存在：{image_path}")
        values = list(boxes)
        for box in values:
            if box.class_id not in self.class_names:
                raise ValueError(f"类别编号不存在：{box.class_id}")
        label = _label_for_image(image_path)
        label.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(box.to_line() for box in values)
        label.write_text(text + ("\n" if text else ""), encoding="utf-8")
        return label

    def import_images(self, paths: Iterable[str | Path], split: str) -> List[Path]:
        self._check_split(split)
        imported: List[Path] = []
        for value in paths:
            source = Path(value).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = self._unique_target(self.image_dirs[split], source.stem, source.suffix.lower())
            shutil.copy2(source, target)
            imported.append(target)
        return imported

    def import_bgr(self, image_bgr: np.ndarray, split: str, stem: str = "camera") -> Path:
        self._check_split(split)
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("当前相机帧无效")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = self._unique_target(self.image_dirs[split], f"{stem}_{timestamp}", ".jpg")
        if not cv2.imwrite(str(target), image_bgr):
            raise RuntimeError(f"图片保存失败：{target}")
        return target

    def set_class_names(self, names: Sequence[str]) -> None:
        cleaned = [str(name).strip() for name in names if str(name).strip()]
        if not cleaned or len(set(cleaned)) != len(cleaned):
            raise ValueError("类别不能为空或重复")
        old = [self.class_names[index] for index in sorted(self.class_names)]
        has_labels = any(
            any(self._label_dir(path).rglob("*.txt")) for path in self.image_dirs.values()
        )
        if has_labels and cleaned[: len(old)] != old:
            raise ValueError("已有标签时不能删除、改名或调整现有类别顺序；只能在末尾新增类别")
        self.raw["names"] = {index: name for index, name in enumerate(cleaned)}
        self.yaml_path.write_text(
            yaml.safe_dump(self.raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        self._reload()

    @staticmethod
    def _unique_target(directory: Path, stem: str, suffix: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stem}{suffix}"
        index = 1
        while target.exists():
            target = directory / f"{stem}_{index}{suffix}"
            index += 1
        return target

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("数据集分组必须是 train 或 val")
