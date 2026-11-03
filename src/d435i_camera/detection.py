"""YOLO adapter and deterministic detectors behind one small interface."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .config import DetectorConfig
from .models import Detection2D, FrameBundle


class DetectorError(RuntimeError):
    pass


def _ensure_lzma_runtime() -> None:
    """Work around Python builds compiled without the optional stdlib _lzma module."""

    try:
        import lzma  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"_lzma", "lzma"}:
            raise
        try:
            from backports import lzma as compatible_lzma
        except ImportError as backport_error:
            raise DetectorError(
                "当前 Python 缺少 _lzma；请安装 backports.lzma 后再加载 YOLO"
            ) from backport_error
        sys.modules.pop("lzma", None)
        sys.modules["lzma"] = compatible_lzma


class ObjectDetector(ABC):
    @property
    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def class_names(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: FrameBundle) -> Tuple[List[Detection2D], float]:
        """Return detections and inference duration in milliseconds."""


class NullDetector(ObjectDetector):
    @property
    def ready(self) -> bool:
        return False

    @property
    def class_names(self) -> Any:
        return {}

    def detect(self, frame: FrameBundle) -> Tuple[List[Detection2D], float]:
        del frame
        return [], 0.0


class SyntheticDetector(ObjectDetector):
    @property
    def ready(self) -> bool:
        return True

    @property
    def class_names(self) -> Any:
        return {39: "bottle", 41: "cup"}

    def detect(self, frame: FrameBundle) -> Tuple[List[Detection2D], float]:
        detections = [
            Detection2D(
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                confidence=float(item["confidence"]),
                bbox_xyxy=tuple(float(value) for value in item["bbox"]),
            )
            for item in frame.metadata.get("mock_objects", [])
        ]
        return detections, 0.2


class YoloDetector(ObjectDetector):
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._model_path = ""
        if config.enabled and config.model_path:
            self.load(config.model_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> Any:
        return getattr(self._model, "names", {}) if self._model is not None else {}

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self, model_path: str) -> None:
        if not model_path:
            raise DetectorError("YOLO 模型路径为空")
        path = Path(model_path).expanduser()
        if path.suffix.lower() in {".pt", ".onnx", ".engine"} and not path.exists():
            raise DetectorError(f"YOLO 模型不存在：{path}")
        _ensure_lzma_runtime()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("未安装 ultralytics；请运行 scripts/install.sh") from exc
        try:
            model = YOLO(str(path if path.exists() else model_path))
            # The remote workstation has no CUDA GPU; fuse is still useful for CPU .pt models.
            try:
                model.fuse()
            except (AttributeError, RuntimeError):
                pass
            self._model = model
            self._model_path = str(path if path.exists() else model_path)
            self.config.model_path = self._model_path
        except Exception as exc:
            raise DetectorError(f"YOLO 模型加载失败：{exc}") from exc

    def unload(self) -> None:
        self._model = None
        self._model_path = ""

    def _class_name(self, class_id: int) -> str:
        names = self.class_names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, Sequence) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def detect(self, frame: FrameBundle) -> Tuple[List[Detection2D], float]:
        if self._model is None or not self.config.enabled:
            return [], 0.0
        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame.color_bgr,
                conf=float(self.config.confidence),
                iou=float(self.config.iou),
                imgsz=int(self.config.image_size),
                device=self.config.device,
                classes=self.config.classes or None,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO 推理失败：{exc}") from exc
        inference_ms = (time.perf_counter() - started) * 1000.0
        detections: List[Detection2D] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            mask_arrays: Optional[np.ndarray] = None
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_arrays = masks.data.cpu().numpy()
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
                track_id: Optional[int] = None
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0].item())
                mask = mask_arrays[index] if mask_arrays is not None and index < len(mask_arrays) else None
                detections.append(
                    Detection2D(
                        class_id=class_id,
                        class_name=self._class_name(class_id),
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                        track_id=track_id,
                        mask=mask,
                    )
                )
        return detections, inference_ms
