"""Public API for the XuK D435i camera module."""

from .annotation import YoloAnnotationProject, YoloBox, initialize_detection_dataset
from .config import AppConfig, load_config, save_config
from .models import (
    CameraDeviceInfo,
    Detection2D,
    FrameBundle,
    ImuSample,
    Intrinsics,
    ObjectObservation,
    PerceptionFrame,
    Pose6D,
    StreamProfile,
)
from .service import CameraModule
from .training import (
    DatasetSummary,
    TrainingConfig,
    TrainingResult,
    YoloTrainingService,
    validate_yolo_dataset,
)

__all__ = [
    "AppConfig",
    "CameraDeviceInfo",
    "CameraModule",
    "Detection2D",
    "DatasetSummary",
    "FrameBundle",
    "ImuSample",
    "Intrinsics",
    "ObjectObservation",
    "PerceptionFrame",
    "Pose6D",
    "StreamProfile",
    "TrainingConfig",
    "TrainingResult",
    "YoloTrainingService",
    "YoloAnnotationProject",
    "YoloBox",
    "initialize_detection_dataset",
    "load_config",
    "save_config",
    "validate_yolo_dataset",
]

__version__ = "0.1.0"
