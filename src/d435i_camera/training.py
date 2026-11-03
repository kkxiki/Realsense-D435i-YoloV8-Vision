"""Validated custom YOLO training interface used by CLI and GUI."""

from __future__ import annotations

import threading
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ProgressCallback = Callable[[Dict[str, Any]], None]


class TrainingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    yaml_path: Path
    root: Path
    task: str
    class_names: Dict[int, str]
    train_images: int
    val_images: int
    label_files: int
    labeled_objects: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "yaml_path": str(self.yaml_path),
            "root": str(self.root),
            "task": self.task,
            "class_names": self.class_names,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "label_files": self.label_files,
            "labeled_objects": self.labeled_objects,
        }


@dataclass(frozen=True)
class TrainingConfig:
    dataset_yaml: str
    base_model: str
    task: str = "detect"
    epochs: int = 100
    image_size: int = 640
    batch: int = 8
    device: str = "cpu"
    workers: int = 4
    patience: int = 30
    project: str = "training_runs"
    name: str = "custom_object"
    cache: bool = False


@dataclass(frozen=True)
class TrainingResult:
    save_dir: Path
    best_model: Path
    last_model: Path
    metrics: Dict[str, float]
    dataset: DatasetSummary

    def as_dict(self) -> Dict[str, Any]:
        return {
            "save_dir": str(self.save_dir),
            "best_model": str(self.best_model),
            "last_model": str(self.last_model),
            "metrics": self.metrics,
            "dataset": self.dataset.as_dict(),
        }


def _class_names(raw: Any) -> Dict[int, str]:
    if isinstance(raw, list):
        names = {index: str(name) for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        names = {int(index): str(name) for index, name in raw.items()}
    else:
        raise ValueError("data.yaml 的 names 必须是列表或编号字典")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("类别编号必须从 0 开始且连续")
    if any(not name.strip() for name in names.values()):
        raise ValueError("类别名称不能为空")
    return names


def _split_paths(root: Path, value: Any) -> List[Path]:
    values: Sequence[Any] = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(str(item)).expanduser()
        paths.append(path if path.is_absolute() else (root / path).resolve())
    return paths


def _images(paths: Iterable[Path]) -> List[Path]:
    output: List[Path] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"数据集路径不存在：{path}")
        if path.is_dir():
            output.extend(
                item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if not value:
                    continue
                image = Path(value).expanduser()
                output.append(image if image.is_absolute() else (path.parent / image).resolve())
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            output.append(path)
    return sorted(set(output))


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if indexes:
        parts[indexes[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def validate_yolo_dataset(dataset_yaml: str | Path, task: str = "detect") -> DatasetSummary:
    task = str(task).strip().lower()
    if task not in {"detect", "segment"}:
        raise ValueError("训练任务必须是 detect 或 segment")
    yaml_path = Path(dataset_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise ValueError(f"数据集 YAML 不存在：{yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = _class_names(raw.get("names"))
    root_value = raw.get("path", ".")
    root = Path(str(root_value)).expanduser()
    root = root if root.is_absolute() else (yaml_path.parent / root).resolve()
    if "train" not in raw or "val" not in raw:
        raise ValueError("data.yaml 必须包含 train 和 val")
    train_images = _images(_split_paths(root, raw["train"]))
    val_images = _images(_split_paths(root, raw["val"]))
    if not train_images:
        raise ValueError("训练集没有图像")
    if not val_images:
        raise ValueError("验证集没有图像")

    label_files = 0
    labeled_objects = 0
    errors: List[str] = []
    for image in train_images + val_images:
        label = _label_for_image(image)
        if not label.exists():
            continue  # A missing label is a valid negative/background image.
        label_files += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            expected = len(fields) == 5 if task == "detect" else len(fields) >= 7 and len(fields) % 2 == 1
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                expected = False
                class_id, coordinates = -1, []
            if not expected or class_id not in names or any(value < 0.0 or value > 1.0 for value in coordinates):
                errors.append(f"{label}:{line_number}")
                if len(errors) >= 10:
                    break
            labeled_objects += 1
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError("标签格式/类别/归一化坐标错误：" + ", ".join(errors))
    if labeled_objects == 0:
        raise ValueError("数据集中没有任何已标注物体")
    return DatasetSummary(
        yaml_path, root, task, names, len(train_images), len(val_images), label_files, labeled_objects
    )


class YoloTrainingService:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def validate(self, config: TrainingConfig) -> DatasetSummary:
        if not 1 <= int(config.epochs) <= 10000:
            raise ValueError("epochs 必须在 1 到 10000 之间")
        if int(config.image_size) < 32 or int(config.image_size) % 32:
            raise ValueError("image_size 必须是不小于 32 的 32 倍数")
        if int(config.batch) == 0 or int(config.batch) < -1:
            raise ValueError("batch 必须为 -1（自动）或正整数")
        model = self._resolve(config.base_model)
        if not model.is_file():
            raise ValueError(f"基础模型不存在：{model}")
        if model.suffix.lower() != ".pt":
            raise ValueError("训练基础模型必须是 .pt 文件")
        if config.task == "segment" and "seg" not in model.stem.lower():
            raise ValueError("segment 任务请选择 yolov8*-seg.pt 基础模型")
        return validate_yolo_dataset(self._resolve(config.dataset_yaml), config.task)

    def train(
        self,
        config: TrainingConfig,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> TrainingResult:
        summary = self.validate(config)
        model_path = self._resolve(config.base_model)
        project = self._resolve(config.project)
        project.mkdir(parents=True, exist_ok=True)
        # This workstation's custom Python 3.10 build has no stdlib _lzma. Reuse
        # the same verified compatibility shim as runtime YOLO inference before
        # torchvision imports lzma during validation.
        from .detection import _ensure_lzma_runtime

        _ensure_lzma_runtime()
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        dataset_configs = project / "_dataset_configs"
        dataset_configs.mkdir(parents=True, exist_ok=True)
        normalized_data = yaml.safe_load(summary.yaml_path.read_text(encoding="utf-8")) or {}
        normalized_data["path"] = str(summary.root)
        digest = hashlib.sha1(str(summary.yaml_path).encode("utf-8")).hexdigest()[:10]
        normalized_yaml = dataset_configs / f"{summary.yaml_path.stem}_{digest}.yaml"
        normalized_yaml.write_text(
            yaml.safe_dump(normalized_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        def check_cancel(_trainer: Any) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise TrainingCancelled("用户已取消训练")

        def epoch_end(trainer: Any) -> None:
            check_cancel(trainer)
            if progress:
                metrics = {
                    str(key): float(value)
                    for key, value in dict(getattr(trainer, "metrics", {}) or {}).items()
                    if isinstance(value, (int, float))
                }
                progress(
                    {
                        "event": "epoch_end",
                        "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                        "epochs": int(config.epochs),
                        "metrics": metrics,
                    }
                )

        model.add_callback("on_train_batch_end", check_cancel)
        model.add_callback("on_train_epoch_end", epoch_end)
        if progress:
            progress({"event": "start", "dataset": summary.as_dict()})
        result = model.train(
            data=str(normalized_yaml),
            task=config.task,
            epochs=int(config.epochs),
            imgsz=int(config.image_size),
            batch=int(config.batch),
            device=str(config.device),
            workers=max(0, int(config.workers)),
            patience=max(0, int(config.patience)),
            project=str(project),
            name=str(config.name).strip() or "custom_object",
            cache=bool(config.cache),
            exist_ok=False,
        )
        save_dir = Path(result.save_dir).resolve()
        metrics = {
            str(key): float(value)
            for key, value in dict(getattr(result, "results_dict", {}) or {}).items()
            if isinstance(value, (int, float))
        }
        training_result = TrainingResult(
            save_dir=save_dir,
            best_model=save_dir / "weights" / "best.pt",
            last_model=save_dir / "weights" / "last.pt",
            metrics=metrics,
            dataset=summary,
        )
        if not training_result.best_model.is_file():
            raise RuntimeError(f"训练结束但未找到 best.pt：{training_result.best_model}")
        if progress:
            progress({"event": "complete", "result": training_result.as_dict()})
        return training_result
