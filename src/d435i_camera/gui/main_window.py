"""Compact operator GUI for live view, calibration, model and device functions."""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..calibration import (
    EyeToHandCalibrationSession,
    HandEyeCalibrationSession,
    IntrinsicCalibrationSession,
)
from ..annotation import YoloAnnotationProject, initialize_detection_dataset
from ..config import save_config
from ..geometry import pose_matrix_from_rpy
from ..labels import chinese_class_name
from ..models import PerceptionFrame
from ..overlay import depth_colormap
from ..robot_pose import ManualRobotPoseProvider
from ..service import CameraModule
from ..training import TrainingConfig, TrainingCancelled, YoloTrainingService
from .annotation_dialog import AnnotationDialog


STYLE = """
QMainWindow, QWidget { background: #171b22; color: #e8edf4; font-size: 13px; }
QFrame#sidebar { background: #11151b; border-right: 1px solid #2b3440; }
QFrame#header { background: #1d232c; border-bottom: 1px solid #303a48; }
QListWidget { background: transparent; border: none; outline: none; padding: 8px; }
QListWidget::item { padding: 13px 12px; margin: 3px; border-radius: 6px; color: #abb7c7; }
QListWidget::item:selected { background: #2467d1; color: white; }
QPushButton { background: #2b3441; border: 1px solid #3b4757; padding: 8px 14px; border-radius: 5px; }
QPushButton:hover { background: #354255; }
QPushButton#primary { background: #2467d1; border-color: #2d78ed; }
QPushButton#danger { background: #8f3541; border-color: #b64b59; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTableWidget {
  background: #11161d; border: 1px solid #354151; border-radius: 4px; padding: 5px;
  selection-background-color: #2467d1;
}
QGroupBox { border: 1px solid #303b48; border-radius: 6px; margin-top: 10px; padding-top: 12px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QTabWidget::pane { border: 1px solid #303b48; }
QTabBar::tab { background: #222a35; padding: 9px 18px; }
QTabBar::tab:selected { background: #2467d1; }
QHeaderView::section { background: #222a35; color: #dce4ef; border: 0; padding: 7px; }
QLabel#video { background: #080b0f; border: 1px solid #303b48; border-radius: 6px; }
QLabel#statusGood { color: #65d98b; font-weight: bold; }
QLabel#statusBad { color: #ff7f8c; font-weight: bold; }
"""


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    status_changed = pyqtSignal(object, str)
    failed = pyqtSignal(object, str)

    def __init__(self, module: CameraModule) -> None:
        super().__init__()
        self.module = module
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        try:
            device = self.module.start()
            self.status_changed.emit(self, f"已连接 {device.name} / {device.serial}")
            target_interval = 1.0 / max(1, self.module.config.camera.fps)
            while not self._stop_event.is_set():
                started = time.perf_counter()
                result = self.module.process_once()
                self.frame_ready.emit(self, result)
                remaining = target_interval - (time.perf_counter() - started)
                if remaining > 0 and self.module.config.camera.backend == "mock":
                    self._stop_event.wait(remaining)
        except Exception as exc:
            self.failed.emit(self, f"{exc}\n{traceback.format_exc()}")
        finally:
            self.module.stop()
            self.status_changed.emit(self, "相机已停止")


class ModelTrainingWorker(QThread):
    progress_changed = pyqtSignal(object, dict)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, service: YoloTrainingService, config: TrainingConfig) -> None:
        super().__init__()
        self.service = service
        self.config = config
        self.cancel_event = threading.Event()

    def stop(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = self.service.train(
                self.config,
                progress=lambda event: self.progress_changed.emit(self, event),
                cancel_event=self.cancel_event,
            )
            self.completed.emit(self, result)
        except TrainingCancelled as exc:
            self.failed.emit(self, str(exc))
        except Exception as exc:
            self.failed.emit(self, f"{exc}\n{traceback.format_exc()}")


class VideoLabel(QLabel):
    def __init__(self, placeholder: str = "相机未启动") -> None:
        super().__init__(placeholder)
        self.setObjectName("video")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self._pixmap: Optional[QPixmap] = None

    def set_bgr(self, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


class MainWindow(QMainWindow):
    def __init__(self, module: CameraModule) -> None:
        super().__init__()
        self.module = module
        # The operator requested per-object labels only; never restore the legacy
        # top-left summary from an older saved configuration.
        self.module.config.gui.show_top_left_panel = False
        if self.module.robot_pose_provider is None:
            self.module.set_robot_pose_provider(ManualRobotPoseProvider())
        self.worker: Optional[CameraWorker] = None
        self.current_result: Optional[PerceptionFrame] = None
        self.intrinsic_session: Optional[IntrinsicCalibrationSession] = None
        self.handeye_session: Optional[HandEyeCalibrationSession] = None
        self.eye_to_hand_session: Optional[EyeToHandCalibrationSession] = None
        self.training_worker: Optional[ModelTrainingWorker] = None
        self.annotation_dialog: Optional[AnnotationDialog] = None
        self.setWindowTitle("XuK D435i 相机与 6D 视觉模块")
        self.resize(1480, 900)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh_configuration()
        self.refresh_stream_profiles(silent=True)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        side_layout = QVBoxLayout(sidebar)
        title = QLabel("XuK VISION")
        title.setFont(QFont("Sans", 16, QFont.Bold))
        title.setStyleSheet("padding: 18px 14px; color: #72a7ff;")
        side_layout.addWidget(title)
        self.nav = QListWidget()
        for text in ["实时检测", "相机功能", "标定中心", "YOLO 模型", "接口与日志"]:
            self.nav.addItem(QListWidgetItem(text))
        self.nav.currentRowChanged.connect(self._change_page)
        side_layout.addWidget(self.nav, 1)
        self.version_label = QLabel("Camera Module v0.1.0")
        self.version_label.setStyleSheet("color:#667386; padding:14px;")
        side_layout.addWidget(self.version_label)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)
        self.page_title = QLabel("实时检测")
        self.page_title.setFont(QFont("Sans", 15, QFont.Bold))
        self.connection_label = QLabel("未连接")
        self.connection_label.setObjectName("statusBad")
        self.metrics_label = QLabel("FPS -- | 延迟 -- ms")
        self.start_button = QPushButton("启动相机")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        header_layout.addWidget(self.page_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.metrics_label)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.connection_label)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)
        content_layout.addWidget(header)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_live_page())
        self.pages.addWidget(self._build_camera_page())
        self.pages.addWidget(self._build_calibration_page())
        self.pages.addWidget(self._build_detector_page())
        self.pages.addWidget(self._build_interface_page())
        content_layout.addWidget(self.pages, 1)

        root.addWidget(sidebar)
        root.addWidget(content, 1)
        self.setCentralWidget(central)
        self.nav.setCurrentRow(0)

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["6D 叠加画面", "原始彩色", "深度伪彩"])
        self.output_frame_combo = QComboBox()
        self.output_frame_combo.addItem(
            "相机坐标系", ("camera_color_optical_frame", "eye_in_hand")
        )
        self.output_frame_combo.addItem("基座坐标系（腕部相机）", ("base_link", "eye_in_hand"))
        self.output_frame_combo.addItem("基座坐标系（头部固定相机）", ("base_link", "eye_to_hand"))
        desired = self.module.config.pose.output_frame
        mount = self.module.config.pose.camera_mount
        selected = 0 if desired == "camera_color_optical_frame" else (2 if mount == "eye_to_hand" else 1)
        self.output_frame_combo.setCurrentIndex(selected)
        self.output_frame_combo.currentIndexChanged.connect(self._change_output_frame)
        snapshot = QPushButton("保存快照")
        snapshot.clicked.connect(self.save_snapshot)
        self.axes_check = QCheckBox("物体坐标轴")
        self.axes_check.setChecked(self.module.config.gui.show_pose_axes)
        self.axes_check.toggled.connect(self._toggle_overlays)
        toolbar.addWidget(QLabel("显示："))
        toolbar.addWidget(self.view_mode)
        toolbar.addWidget(QLabel("6D 坐标系："))
        toolbar.addWidget(self.output_frame_combo)
        toolbar.addWidget(self.axes_check)
        toolbar.addStretch(1)
        toolbar.addWidget(snapshot)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.live_video = VideoLabel()
        splitter.addWidget(self.live_video)
        details = QWidget()
        details.setMinimumWidth(390)
        details_layout = QVBoxLayout(details)
        details_layout.addWidget(QLabel("实时物体 6D 信息"))
        self.object_table = QTableWidget(0, 8)
        self.object_table.setHorizontalHeaderLabels(
            ["物品中文名称", "置信度", "X mm", "Y mm", "Z mm", "Rx°", "Ry°", "Rz°"]
        )
        self.object_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.object_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.object_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        details_layout.addWidget(self.object_table, 1)
        self.frame_info = QTextEdit()
        self.frame_info.setReadOnly(True)
        self.frame_info.setMaximumHeight(170)
        details_layout.addWidget(self.frame_info)
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        return page

    def _build_camera_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        stream_group = QGroupBox("视频流参数（RGB 与 Depth 独立配置）")
        stream_layout = QGridLayout(stream_group)
        self.color_profile_combo = QComboBox()
        self.color_profile_combo.setMinimumWidth(280)
        self.depth_profile_combo = QComboBox()
        self.depth_profile_combo.setMinimumWidth(280)
        self.imu_check = QCheckBox("启用 D435i IMU")
        self.imu_check.setChecked(self.module.config.camera.enable_imu)
        refresh_profiles = QPushButton("刷新支持列表")
        refresh_profiles.clicked.connect(self.refresh_stream_profiles)
        apply_profile = QPushButton("应用并重启相机")
        apply_profile.setObjectName("primary")
        apply_profile.clicked.connect(self.apply_stream_profile)
        self.stream_status = QLabel("尚未读取支持的分辨率/帧率")
        stream_layout.addWidget(QLabel("RGB 彩色分辨率 / 帧率："), 0, 0)
        stream_layout.addWidget(self.color_profile_combo, 0, 1)
        stream_layout.addWidget(QLabel("Depth 深度分辨率 / 帧率："), 1, 0)
        stream_layout.addWidget(self.depth_profile_combo, 1, 1)
        stream_layout.addWidget(self.imu_check, 0, 2)
        stream_layout.addWidget(refresh_profiles, 1, 2)
        stream_layout.addWidget(apply_profile, 0, 3, 2, 1)
        stream_layout.addWidget(self.stream_status, 2, 0, 1, 4)
        stream_layout.setColumnStretch(4, 1)
        layout.addWidget(stream_group)

        device_group = QGroupBox("D435i 设备信息")
        device_layout = QVBoxLayout(device_group)
        buttons = QHBoxLayout()
        refresh = QPushButton("枚举设备")
        refresh.clicked.connect(self.refresh_devices)
        export = QPushButton("导出 RealSense 出厂内参")
        export.clicked.connect(self.export_factory_intrinsics)
        export_ply = QPushButton("保存当前点云 PLY")
        export_ply.clicked.connect(self.export_point_cloud)
        buttons.addWidget(refresh)
        buttons.addWidget(export)
        buttons.addWidget(export_ply)
        buttons.addStretch(1)
        device_layout.addLayout(buttons)
        self.device_table = QTableWidget(0, 6)
        self.device_table.setHorizontalHeaderLabels(
            ["名称", "序列号", "固件", "USB", "产品线", "IMU"]
        )
        self.device_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.device_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        device_layout.addWidget(self.device_table)
        layout.addWidget(device_group)

        controls_group = QGroupBox("相机参数（启动相机后刷新）")
        controls_layout = QVBoxLayout(controls_group)
        controls_toolbar = QHBoxLayout()
        refresh_controls = QPushButton("刷新参数")
        refresh_controls.clicked.connect(self.refresh_controls)
        self.control_filter = QComboBox()
        self.control_filter.addItem("全部参数", "")
        self.control_filter.addItem("RGB 彩色参数（color.*）", "color.")
        self.control_filter.addItem("Depth 深度参数（depth.*）", "depth.")
        self.control_filter.addItem("Motion/IMU 参数（motion.*）", "motion.")
        self.control_filter.currentIndexChanged.connect(self.refresh_controls)
        controls_toolbar.addWidget(refresh_controls)
        controls_toolbar.addWidget(QLabel("参数分组："))
        controls_toolbar.addWidget(self.control_filter)
        controls_toolbar.addStretch(1)
        controls_layout.addLayout(controls_toolbar)
        self.controls_table = QTableWidget(0, 8)
        self.controls_table.setHorizontalHeaderLabels(
            ["参数", "当前值", "最小", "最大", "步长", "传感器", "状态", "设置"]
        )
        self.controls_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.controls_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        controls_layout.addWidget(self.controls_table)
        layout.addWidget(controls_group, 1)
        return page

    def _build_calibration_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        board = self.module.config.calibration.board
        board_group = QGroupBox("标定板参数（用户已确认）")
        board_form = QFormLayout(board_group)
        board_form.addRow("棋盘内角点", QLabel(f"{board.inner_corners_cols} × {board.inner_corners_rows}"))
        board_form.addRow("单格边长", QLabel(f"{board.square_size_mm:.3f} mm"))
        board_form.addRow("状态", QLabel("已确认" if board.confirmed else "未确认，禁止求解"))
        layout.addWidget(board_group)

        tabs = QTabWidget()
        tabs.addTab(self._build_intrinsic_tab(), "OpenCV 内参标定")
        tabs.addTab(self._build_handeye_tab(), "眼在手上手眼标定")
        tabs.addTab(self._build_eye_to_hand_tab(), "头部相机眼在手外标定")
        layout.addWidget(tabs, 1)
        return page

    def _build_intrinsic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        controls = QVBoxLayout()
        description = QLabel(
            "从不同距离、位置和倾角采集棋盘。重复视角会被拒绝。\n"
            f"至少 {self.module.config.calibration.min_intrinsic_samples} 帧后才能求解。"
        )
        description.setWordWrap(True)
        self.intrinsic_count = QLabel("样本：0")
        reset = QPushButton("新建/重置会话")
        capture = QPushButton("采集当前帧")
        solve = QPushButton("求解并保存")
        solve.setObjectName("primary")
        reset.clicked.connect(self.reset_intrinsic)
        capture.clicked.connect(self.capture_intrinsic)
        solve.clicked.connect(self.solve_intrinsic)
        self.intrinsic_status = QTextEdit()
        self.intrinsic_status.setReadOnly(True)
        controls.addWidget(description)
        controls.addWidget(self.intrinsic_count)
        controls.addWidget(reset)
        controls.addWidget(capture)
        controls.addWidget(solve)
        controls.addWidget(self.intrinsic_status, 1)
        left = QWidget()
        left.setLayout(controls)
        left.setMaximumWidth(390)
        self.intrinsic_preview = VideoLabel("等待采集棋盘")
        layout.addWidget(left)
        layout.addWidget(self.intrinsic_preview, 1)
        return tab

    def _build_handeye_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setMaximumWidth(430)
        controls = QVBoxLayout(left)
        note = QLabel(
            "相机固定在末端，棋盘固定在机器人基座。每次机械臂静止后采样。\n"
            "ROS 位姿接入完成后会自动使用订阅值；当前也可手动输入验证。"
        )
        note.setWordWrap(True)
        controls.addWidget(note)
        pose_box = QGroupBox("base_T_tool 当前位姿")
        grid = QGridLayout(pose_box)
        self.pose_inputs = []
        labels = ["X mm", "Y mm", "Z mm", "Rx °", "Ry °", "Rz °"]
        for index, label in enumerate(labels):
            spin = QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)
            grid.addWidget(QLabel(label), index // 3 * 2, index % 3)
            grid.addWidget(spin, index // 3 * 2 + 1, index % 3)
            self.pose_inputs.append(spin)
        controls.addWidget(pose_box)
        self.handeye_count = QLabel("样本：0")
        controls.addWidget(self.handeye_count)
        reset = QPushButton("新建/重置会话")
        capture = QPushButton("读取 ROS/输入位姿并采样")
        solve = QPushButton("求解 tool_T_camera 并保存")
        solve.setObjectName("primary")
        reset.clicked.connect(self.reset_handeye)
        capture.clicked.connect(self.capture_handeye)
        solve.clicked.connect(self.solve_handeye)
        controls.addWidget(reset)
        controls.addWidget(capture)
        controls.addWidget(solve)
        self.handeye_status = QTextEdit()
        self.handeye_status.setReadOnly(True)
        controls.addWidget(self.handeye_status, 1)
        self.handeye_preview = VideoLabel("等待采集棋盘和末端位姿")
        layout.addWidget(left)
        layout.addWidget(self.handeye_preview, 1)
        return tab

    def _build_eye_to_hand_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setMaximumWidth(470)
        controls = QVBoxLayout(left)
        note = QLabel(
            "相机固定在机器人头部/基座，棋盘刚性固定在末端工具上。"
            "每次机械臂静止后，同步采集 base_T_tool 与棋盘图像。\n"
            "求解输出 base_T_camera。若头部关节会运动，请标定时保持头部固定，"
            "运行时由机械臂侧提供实时 base_T_head 变换。"
        )
        note.setWordWrap(True)
        controls.addWidget(note)
        pose_box = QGroupBox("base_T_tool 当前位姿")
        grid = QGridLayout(pose_box)
        self.eye_to_hand_pose_inputs = []
        labels = ["X mm", "Y mm", "Z mm", "Rx °", "Ry °", "Rz °"]
        for index, label in enumerate(labels):
            spin = QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)
            grid.addWidget(QLabel(label), index // 3 * 2, index % 3)
            grid.addWidget(spin, index // 3 * 2 + 1, index % 3)
            self.eye_to_hand_pose_inputs.append(spin)
        controls.addWidget(pose_box)
        self.eye_to_hand_count = QLabel("样本：0")
        controls.addWidget(self.eye_to_hand_count)
        reset = QPushButton("新建/重置会话")
        capture = QPushButton("读取 ROS/输入位姿并采样")
        solve = QPushButton("求解 base_T_camera 并保存")
        solve.setObjectName("primary")
        reset.clicked.connect(self.reset_eye_to_hand)
        capture.clicked.connect(self.capture_eye_to_hand)
        solve.clicked.connect(self.solve_eye_to_hand)
        controls.addWidget(reset)
        controls.addWidget(capture)
        controls.addWidget(solve)
        self.eye_to_hand_status = QTextEdit()
        self.eye_to_hand_status.setReadOnly(True)
        controls.addWidget(self.eye_to_hand_status, 1)
        self.eye_to_hand_preview = VideoLabel("等待采集末端棋盘和机器人位姿")
        layout.addWidget(left)
        layout.addWidget(self.eye_to_hand_preview, 1)
        return tab

    def _build_detector_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("YOLOv8 模型")
        form = QFormLayout(group)
        path_row = QHBoxLayout()
        self.model_path = QLineEdit(self.module.config.detector.model_path)
        browse = QPushButton("选择…")
        browse.clicked.connect(self.browse_model)
        path_row.addWidget(self.model_path, 1)
        path_row.addWidget(browse)
        form.addRow("模型路径", path_row)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(self.module.config.detector.confidence)
        form.addRow("置信度", self.confidence)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "0"])
        self.device_combo.setCurrentText(self.module.config.detector.device)
        form.addRow("推理设备", self.device_combo)
        load = QPushButton("加载/热切换模型")
        load.setObjectName("primary")
        load.clicked.connect(self.load_model)
        form.addRow("", load)
        layout.addWidget(group)

        training_group = QGroupBox("训练新的物品模型")
        training_form = QFormLayout(training_group)
        dataset_row = QHBoxLayout()
        self.training_dataset = QLineEdit("datasets/custom/data.yaml")
        choose_dataset = QPushButton("选择…")
        choose_dataset.clicked.connect(self.browse_training_dataset)
        dataset_row.addWidget(self.training_dataset, 1)
        dataset_row.addWidget(choose_dataset)
        training_form.addRow("数据集 data.yaml", dataset_row)
        self.training_task = QComboBox()
        self.training_task.addItem("目标检测（推荐）", "detect")
        self.training_task.addItem("实例分割（轮廓更准）", "segment")
        training_form.addRow("任务", self.training_task)
        training_options = QHBoxLayout()
        self.training_epochs = QSpinBox()
        self.training_epochs.setRange(1, 10000)
        self.training_epochs.setValue(100)
        self.training_image_size = QSpinBox()
        self.training_image_size.setRange(32, 2048)
        self.training_image_size.setSingleStep(32)
        self.training_image_size.setValue(640)
        self.training_batch = QSpinBox()
        self.training_batch.setRange(-1, 1024)
        self.training_batch.setValue(8)
        training_options.addWidget(QLabel("轮数"))
        training_options.addWidget(self.training_epochs)
        training_options.addWidget(QLabel("图像尺寸"))
        training_options.addWidget(self.training_image_size)
        training_options.addWidget(QLabel("批次"))
        training_options.addWidget(self.training_batch)
        training_options.addStretch(1)
        training_form.addRow("训练参数", training_options)
        self.training_device = QComboBox()
        self.training_device.addItems(["cpu", "0"])
        self.training_name = QLineEdit("custom_object")
        training_form.addRow("训练设备", self.training_device)
        training_form.addRow("任务名称", self.training_name)
        training_buttons = QHBoxLayout()
        annotate_training = QPushButton("可视化标注图片")
        annotate_training.clicked.connect(self.open_annotation_tool)
        validate_training = QPushButton("校验数据集")
        self.start_training_button = QPushButton("开始训练")
        self.start_training_button.setObjectName("primary")
        self.cancel_training_button = QPushButton("取消训练")
        self.cancel_training_button.setObjectName("danger")
        self.cancel_training_button.setEnabled(False)
        validate_training.clicked.connect(self.validate_training_dataset)
        self.start_training_button.clicked.connect(self.start_training)
        self.cancel_training_button.clicked.connect(self.cancel_training)
        training_buttons.addWidget(annotate_training)
        training_buttons.addWidget(validate_training)
        training_buttons.addWidget(self.start_training_button)
        training_buttons.addWidget(self.cancel_training_button)
        training_buttons.addStretch(1)
        training_form.addRow("", training_buttons)
        layout.addWidget(training_group)
        self.training_status = QTextEdit()
        self.training_status.setReadOnly(True)
        self.training_status.setMaximumHeight(170)
        self.training_status.setPlaceholderText("训练开始前会检查图像、标签和类别编号。")
        layout.addWidget(self.training_status)
        self.model_status = QTextEdit()
        self.model_status.setReadOnly(True)
        self.model_status.setText(
            "默认接入 COCO YOLOv8n，已启用全部 80 类，包括人、水瓶、杯子、手机等。\n"
            "普通检测模型采用深度点云 PCA 估计 Rx/Ry/Rz；分割模型会优先使用 mask 内点。"
        )
        layout.addWidget(self.model_status, 1)
        return page

    def _build_interface_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("模块接口状态")
        group_layout = QVBoxLayout(group)
        refresh = QPushButton("刷新状态")
        refresh.clicked.connect(self._refresh_configuration)
        group_layout.addWidget(refresh, 0, Qt.AlignLeft)
        self.health_text = QTextEdit()
        self.health_text.setReadOnly(True)
        group_layout.addWidget(self.health_text)
        layout.addWidget(group, 1)
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group, 1)
        return page

    def _change_page(self, index: int) -> None:
        titles = ["实时检测", "相机功能", "标定中心", "YOLO 模型", "接口与日志"]
        self.pages.setCurrentIndex(max(0, index))
        self.page_title.setText(titles[max(0, index)])

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{stamp}] {message}")

    def start_stream(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.worker = CameraWorker(self.module)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self._on_status)
        self.worker.failed.connect(self._on_failure)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._log("正在启动相机…")

    def stop_stream(self, wait: bool = False) -> None:
        worker = self.worker
        if worker and worker.isRunning():
            worker.stop()
            self.connection_label.setText("正在停止…")
            self.connection_label.setObjectName("statusBad")
            self.connection_label.style().unpolish(self.connection_label)
            self.connection_label.style().polish(self.connection_label)
            self._log("正在停止相机…")
            if wait and not worker.wait(5000):
                self._log("相机线程仍在退出，请稍候")
        self.start_button.setEnabled(not bool(worker and worker.isRunning()))
        self.stop_button.setEnabled(False)

    def _on_status(self, worker: CameraWorker, status: str) -> None:
        if worker is not self.worker:
            return
        self.connection_label.setText(status)
        good = status.startswith("已连接")
        self.connection_label.setObjectName("statusGood" if good else "statusBad")
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)
        self._log(status)
        if not good:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _on_failure(self, worker: CameraWorker, details: str) -> None:
        if worker is not self.worker:
            return
        self._log(details)
        QMessageBox.critical(self, "相机模块错误", details.split("\n", 1)[0])

    def _on_frame(self, worker: CameraWorker, result: PerceptionFrame) -> None:
        if worker is not self.worker:
            return
        self.current_result = result
        mode = self.view_mode.currentIndex()
        if mode == 1:
            image = result.frame.color_bgr
        elif mode == 2 and result.frame.depth_m is not None:
            image = depth_colormap(
                result.frame.depth_m,
                self.module.config.depth.min_m,
                self.module.config.depth.max_m,
            )
        else:
            image = result.annotated_bgr
        self.live_video.set_bgr(image)
        fps = 1000.0 / result.processing_ms if result.processing_ms > 0 else 0.0
        self.metrics_label.setText(
            f"FPS {fps:.1f} | 总延迟 {result.processing_ms:.1f} ms | 推理 {result.inference_ms:.1f} ms"
        )
        self._update_objects(result)

    def _update_objects(self, result: PerceptionFrame) -> None:
        self.object_table.setRowCount(len(result.observations))
        for row, observation in enumerate(result.observations):
            detection = observation.detection
            values = [
                chinese_class_name(detection.class_name),
                f"{detection.confidence * 100.0:.1f}%",
            ]
            if observation.pose:
                values.extend(f"{value * 1000.0:.1f}" for value in observation.pose.translation_m)
                values.extend(f"{value:.1f}" for value in observation.pose.rpy_deg)
            else:
                values.extend(["--"] * 6)
            for column, value in enumerate(values):
                self.object_table.setItem(row, column, QTableWidgetItem(value))
        self.frame_info.setPlainText(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))

    def _toggle_overlays(self) -> None:
        self.module.config.gui.show_top_left_panel = False
        self.module.config.gui.show_pose_axes = self.axes_check.isChecked()

    def _change_output_frame(self) -> None:
        values = self.output_frame_combo.currentData()
        if not values:
            return
        output_frame, mount = values
        self.module.config.pose.output_frame = str(output_frame)
        self.module.config.pose.camera_mount = str(mount)
        try:
            save_config(self.module.config)
            self._log(f"6D 输出已切换：frame={output_frame}, mount={mount}")
        except Exception as exc:
            self._log(f"保存 6D 输出配置失败：{exc}")

    def save_snapshot(self) -> None:
        try:
            paths = self.module.capture_snapshot()
            self._log(f"快照已保存：{paths.color}")
            QMessageBox.information(self, "保存成功", f"彩色图：{paths.color}\n元数据：{paths.metadata}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def refresh_devices(self) -> None:
        try:
            devices = self.module.list_devices()
            self.device_table.setRowCount(len(devices))
            for row, item in enumerate(devices):
                values = [
                    item.name,
                    item.serial,
                    item.firmware,
                    item.usb_type,
                    item.product_line,
                    "是" if item.supports_imu else "否",
                ]
                for column, value in enumerate(values):
                    self.device_table.setItem(row, column, QTableWidgetItem(value))
            self._log(f"枚举到 {len(devices)} 台相机")
            self.refresh_stream_profiles(silent=True)
        except Exception as exc:
            QMessageBox.warning(self, "设备枚举失败", str(exc))

    def refresh_stream_profiles(self, _checked: bool = False, silent: bool = False) -> None:
        try:
            color_profiles = self.module.list_color_profiles()
            depth_profiles = self.module.list_depth_profiles()
            current_color = (
                self.module.config.camera.width,
                self.module.config.camera.height,
                self.module.config.camera.fps,
            )
            current_depth = (
                self.module.config.camera.depth_width,
                self.module.config.camera.depth_height,
                self.module.config.camera.depth_fps,
            )

            def fill(combo, profiles, current):  # type: ignore[no-untyped-def]
                combo.clear()
                selected_index = -1
                for index, profile in enumerate(profiles):
                    values = (profile.width, profile.height, profile.fps)
                    combo.addItem(profile.label, values)
                    if values == current:
                        selected_index = index
                if selected_index >= 0:
                    combo.setCurrentIndex(selected_index)

            fill(self.color_profile_combo, color_profiles, current_color)
            fill(self.depth_profile_combo, depth_profiles, current_depth)
            self.stream_status.setText(
                f"当前 RGB {current_color[0]}×{current_color[1]}@{current_color[2]}；"
                f"Depth {current_depth[0]}×{current_depth[1]}@{current_depth[2]}；"
                f"支持 RGB {len(color_profiles)} 组 / Depth {len(depth_profiles)} 组"
            )
            self._log(
                f"读取到 RGB {len(color_profiles)} 组、Depth {len(depth_profiles)} 组 profile"
            )
        except Exception as exc:
            self.stream_status.setText(f"读取失败：{exc}")
            if not silent:
                QMessageBox.warning(self, "读取视频流配置失败", str(exc))

    def apply_stream_profile(self) -> None:
        color_values = self.color_profile_combo.currentData()
        depth_values = self.depth_profile_combo.currentData()
        if not color_values or not depth_values:
            QMessageBox.warning(self, "没有可用配置", "请先点击“刷新支持列表”")
            return
        color_width, color_height, color_fps = [int(value) for value in color_values]
        depth_width, depth_height, depth_fps = [int(value) for value in depth_values]
        was_running = bool(self.worker and self.worker.isRunning())
        if was_running:
            self.stop_stream(wait=True)
        try:
            result = self.module.configure_camera(
                color_width=color_width,
                color_height=color_height,
                color_fps=color_fps,
                depth_width=depth_width,
                depth_height=depth_height,
                depth_fps=depth_fps,
                enable_imu=self.imu_check.isChecked(),
                restart_if_running=False,
                persist=True,
            )
            self.stream_status.setText(
                f"已应用 RGB {color_width}×{color_height}@{color_fps}，"
                f"Depth {depth_width}×{depth_height}@{depth_fps}，IMU="
                f"{'开' if self.imu_check.isChecked() else '关'}"
            )
            self._log(
                f"视频流配置已应用并保存：RGB {color_width}×{color_height}@{color_fps}，"
                f"Depth {depth_width}×{depth_height}@{depth_fps}，"
                f"IMU={self.imu_check.isChecked()}"
            )
            if was_running:
                self.start_stream()
            QMessageBox.information(
                self,
                "相机参数已修改",
                f"RGB：{color_width}×{color_height} @ {color_fps} FPS\n"
                f"Depth：{depth_width}×{depth_height} @ {depth_fps} FPS\n"
                f"IMU：{'启用' if self.imu_check.isChecked() else '关闭'}\n"
                f"配置文件：{result.get('saved_config')}",
            )
        except Exception as exc:
            if was_running:
                self.start_stream()
            QMessageBox.warning(self, "应用视频流参数失败", str(exc))

    def refresh_controls(self, _value: object = None) -> None:
        try:
            controls = self.module.get_controls()
            prefix = str(self.control_filter.currentData() or "")
            items = [
                (name, item)
                for name, item in controls.items()
                if not prefix or name.startswith(prefix)
            ]
            group_order = {"color": 0, "depth": 1, "motion": 2}
            items.sort(key=lambda pair: (group_order.get(pair[0].split(".", 1)[0], 9), pair[0]))
            self.controls_table.setRowCount(len(items))
            for row, (name, item) in enumerate(items):
                values = [
                    name,
                    item["value"],
                    item["min"],
                    item["max"],
                    item["step"],
                    item["sensor"],
                    "保护项" if item.get("protected") else "只读" if item["read_only"] else "可调",
                ]
                for column, value in enumerate(values):
                    self.controls_table.setItem(row, column, QTableWidgetItem(str(value)))
                editor = QDoubleSpinBox()
                editor.setRange(float(item["min"]), float(item["max"]))
                editor.setSingleStep(max(float(item["step"]), 0.001))
                editor.setValue(float(item["value"]))
                editor.setEnabled(not bool(item["read_only"]))
                editor.editingFinished.connect(
                    lambda n=name, widget=editor: self._set_control(n, widget.value())
                )
                if item.get("protected"):
                    editor.setToolTip("会改变同步、深度单位或固件时序，普通操作界面禁止修改")
                self.controls_table.setCellWidget(row, 7, editor)
        except Exception as exc:
            QMessageBox.warning(self, "读取参数失败", str(exc))

    def _set_control(self, name: str, value: float) -> None:
        try:
            self.module.set_control(name, value)
            self._log(f"设置 {name}={value}")
        except Exception as exc:
            QMessageBox.warning(self, "设置失败", str(exc))

    def export_factory_intrinsics(self) -> None:
        try:
            path = self.module.export_factory_intrinsics()
            QMessageBox.information(self, "导出成功", str(path))
            self._log(f"导出相机内参：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def export_point_cloud(self) -> None:
        try:
            default = self.module.config.resolve(self.module.config.paths.capture_dir) / "pointcloud.ply"
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存点云", str(default), "PLY 点云 (*.ply)"
            )
            if not filename:
                return
            path = self.module.save_point_cloud_ply(filename)
            self._log(f"点云已保存：{path}")
            QMessageBox.information(self, "保存成功", str(path))
        except Exception as exc:
            QMessageBox.warning(self, "点云保存失败", str(exc))

    def reset_intrinsic(self) -> None:
        try:
            self.intrinsic_session = self.module.new_intrinsic_session()
            self.intrinsic_count.setText("样本：0")
            self.intrinsic_status.setText("内参标定会话已新建")
        except Exception as exc:
            QMessageBox.warning(self, "无法新建会话", str(exc))

    def capture_intrinsic(self) -> None:
        try:
            if self.intrinsic_session is None:
                self.reset_intrinsic()
            assert self.intrinsic_session is not None
            frame = self.module.latest_raw
            if frame is None:
                raise ValueError("请先启动相机")
            accepted, message, preview = self.intrinsic_session.add_frame(frame.color_bgr)
            self.intrinsic_preview.set_bgr(preview)
            self.intrinsic_count.setText(f"样本：{self.intrinsic_session.sample_count}")
            self.intrinsic_status.append(message)
            self._log(message)
            if not accepted:
                QApplication.beep()
        except Exception as exc:
            QMessageBox.warning(self, "采集失败", str(exc))

    def solve_intrinsic(self) -> None:
        try:
            if self.intrinsic_session is None:
                raise ValueError("尚未新建内参标定会话")
            result = self.intrinsic_session.solve()
            path = result.save(
                self.module.config.resolve(self.module.config.calibration.intrinsic_file)
            )
            message = f"内参已保存：{path}\n总体 RMS={result.rms_px:.4f}px"
            self.intrinsic_status.append(message)
            QMessageBox.information(self, "内参求解完成", message)
        except Exception as exc:
            QMessageBox.warning(self, "内参求解失败", str(exc))

    def reset_handeye(self) -> None:
        try:
            self.handeye_session = self.module.new_handeye_session()
            self.handeye_count.setText("样本：0")
            self.handeye_status.setText("眼在手上标定会话已新建")
        except Exception as exc:
            QMessageBox.warning(self, "无法新建会话", str(exc))

    def capture_handeye(self) -> None:
        try:
            if self.handeye_session is None:
                self.reset_handeye()
            assert self.handeye_session is not None
            frame = self.module.latest_raw
            if frame is None:
                raise ValueError("请先启动相机")
            timed_pose = self._robot_pose_for_capture(self.pose_inputs)
            accepted, message, preview = self.handeye_session.add_frame(
                frame.color_bgr, timed_pose.base_T_tool, frame.timestamp_s
            )
            self.handeye_preview.set_bgr(preview)
            self.handeye_count.setText(f"样本：{self.handeye_session.sample_count}")
            self.handeye_status.append(f"[{timed_pose.source}] {message}")
            self._log(message)
            if not accepted:
                QApplication.beep()
        except Exception as exc:
            QMessageBox.warning(self, "采集失败", str(exc))

    def solve_handeye(self) -> None:
        try:
            if self.handeye_session is None:
                raise ValueError("尚未新建手眼标定会话")
            result = self.handeye_session.solve()
            path = result.save(
                self.module.config.resolve(self.module.config.calibration.handeye_file)
            )
            self.module.set_tool_T_camera(result.tool_T_camera)
            message = (
                f"tool_T_camera 已保存：{path}\n"
                f"平移 RMS={result.translation_rms_mm:.3f} mm，"
                f"旋转 RMS={result.rotation_rms_deg:.3f}°，"
                f"重投影 RMS={result.mean_reprojection_rms_px:.3f} px"
            )
            self.handeye_status.append(message)
            QMessageBox.information(self, "手眼求解完成", message)
        except Exception as exc:
            QMessageBox.warning(self, "手眼求解失败", str(exc))

    def _robot_pose_for_capture(self, inputs):  # type: ignore[no-untyped-def]
        provider = self.module.robot_pose_provider
        if isinstance(provider, ManualRobotPoseProvider):
            values = [item.value() for item in inputs]
            base_T_tool = pose_matrix_from_rpy(values[:3], values[3:], 0.001)
            return provider.update_matrix(base_T_tool)
        timed_pose = provider.latest(0.5) if provider else None
        if timed_pose is None:
            raise ValueError("未收到 0.5 秒内的 ROS base_T_tool 位姿")
        return timed_pose

    def reset_eye_to_hand(self) -> None:
        try:
            self.eye_to_hand_session = self.module.new_eye_to_hand_session()
            self.eye_to_hand_count.setText("样本：0")
            self.eye_to_hand_status.setText("头部固定相机眼在手外标定会话已新建")
        except Exception as exc:
            QMessageBox.warning(self, "无法新建会话", str(exc))

    def capture_eye_to_hand(self) -> None:
        try:
            if self.eye_to_hand_session is None:
                self.reset_eye_to_hand()
            assert self.eye_to_hand_session is not None
            frame = self.module.latest_raw
            if frame is None:
                raise ValueError("请先启动相机")
            timed_pose = self._robot_pose_for_capture(self.eye_to_hand_pose_inputs)
            accepted, message, preview = self.eye_to_hand_session.add_frame(
                frame.color_bgr, timed_pose.base_T_tool, frame.timestamp_s
            )
            self.eye_to_hand_preview.set_bgr(preview)
            self.eye_to_hand_count.setText(f"样本：{self.eye_to_hand_session.sample_count}")
            self.eye_to_hand_status.append(f"[{timed_pose.source}] {message}")
            self._log(message)
            if not accepted:
                QApplication.beep()
        except Exception as exc:
            QMessageBox.warning(self, "采集失败", str(exc))

    def solve_eye_to_hand(self) -> None:
        try:
            if self.eye_to_hand_session is None:
                raise ValueError("尚未新建头部相机眼在手外标定会话")
            result = self.eye_to_hand_session.solve()
            path = result.save(
                self.module.config.resolve(self.module.config.calibration.eye_to_hand_file)
            )
            self.module.set_base_T_camera(result.base_T_camera)
            message = (
                f"base_T_camera 已保存：{path}\n"
                f"方程残差：平移 RMS={result.translation_rms_mm:.3f} mm，"
                f"旋转 RMS={result.rotation_rms_deg:.3f}°，"
                f"重投影 RMS={result.mean_reprojection_rms_px:.3f} px"
            )
            self.eye_to_hand_status.append(message)
            QMessageBox.information(self, "头部相机手眼求解完成", message)
        except Exception as exc:
            QMessageBox.warning(self, "头部相机手眼求解失败", str(exc))

    def browse_model(self) -> None:
        current = self.model_path.text().strip()
        selected = self.module.config.resolve(current) if current else self.module.config.project_root / "models"
        initial = selected if selected.is_dir() else selected.parent
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", str(initial), "YOLO (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if filename:
            self.model_path.setText(filename)

    def browse_training_dataset(self) -> None:
        current = self.module.config.resolve(self.training_dataset.text().strip())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 数据集 YAML",
            str(current.parent),
            "YAML (*.yaml *.yml);;所有文件 (*)",
        )
        if filename:
            self.training_dataset.setText(filename)

    def _current_color_frame(self) -> Optional[np.ndarray]:
        if self.current_result is None:
            return None
        return self.current_result.frame.color_bgr

    def open_annotation_tool(self) -> None:
        """Open the mouse annotation flow for the selected training dataset."""
        dataset_text = self.training_dataset.text().strip() or "datasets/custom/data.yaml"
        dataset_path = self.module.config.resolve(dataset_text)
        try:
            if not dataset_path.exists():
                initialize_detection_dataset(dataset_path, ["new_object"])
            project = YoloAnnotationProject(dataset_path)
        except Exception as exc:
            QMessageBox.warning(self, "无法打开标注工具", str(exc))
            return
        if self.training_task.currentData() != "detect":
            QMessageBox.information(
                self,
                "检测框标注",
                "当前工具生成YOLO目标检测矩形框。实例分割需要多边形轮廓，请将训练任务切换为“目标检测”。",
            )
            self.training_task.setCurrentIndex(0)
        self.annotation_dialog = AnnotationDialog(
            project=project,
            current_frame=self._current_color_frame,
            parent=self,
        )
        self.annotation_dialog.show()
        self.annotation_dialog.raise_()
        self.annotation_dialog.activateWindow()

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            dataset_yaml=self.training_dataset.text().strip(),
            base_model=self.model_path.text().strip() or "models/yolov8n.pt",
            task=str(self.training_task.currentData()),
            epochs=self.training_epochs.value(),
            image_size=self.training_image_size.value(),
            batch=self.training_batch.value(),
            device=self.training_device.currentText(),
            project="training_runs",
            name=self.training_name.text().strip() or "custom_object",
        )

    def validate_training_dataset(self) -> None:
        try:
            service = YoloTrainingService(self.module.config.project_root)
            summary = service.validate(self._training_config())
            self.training_status.setPlainText(
                "数据集校验通过：\n" + json.dumps(summary.as_dict(), ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            self.training_status.setPlainText(str(exc))
            QMessageBox.warning(self, "数据集校验失败", str(exc))

    def start_training(self) -> None:
        if self.training_worker and self.training_worker.isRunning():
            return
        try:
            service = YoloTrainingService(self.module.config.project_root)
            config = self._training_config()
            summary = service.validate(config)
        except Exception as exc:
            QMessageBox.warning(self, "无法开始训练", str(exc))
            return
        if self.worker and self.worker.isRunning():
            self.stop_stream(wait=True)
        worker = ModelTrainingWorker(service, config)
        self.training_worker = worker
        worker.progress_changed.connect(self._on_training_progress)
        worker.completed.connect(self._on_training_completed)
        worker.failed.connect(self._on_training_failed)
        self.start_training_button.setEnabled(False)
        self.cancel_training_button.setEnabled(True)
        self.training_status.setPlainText(
            "训练已启动，相机已释放。\n" + json.dumps(summary.as_dict(), ensure_ascii=False, indent=2)
        )
        worker.start()

    def cancel_training(self) -> None:
        worker = self.training_worker
        if worker and worker.isRunning():
            worker.stop()
            self.cancel_training_button.setEnabled(False)
            self.training_status.append("正在取消训练…")

    def _on_training_progress(self, worker: ModelTrainingWorker, event: dict) -> None:
        if worker is not self.training_worker:
            return
        if event.get("event") == "epoch_end":
            self.training_status.append(
                f"Epoch {event.get('epoch')}/{event.get('epochs')}  "
                + json.dumps(event.get("metrics", {}), ensure_ascii=False)
            )

    def _on_training_completed(self, worker: ModelTrainingWorker, result: object) -> None:
        if worker is not self.training_worker:
            return
        self.start_training_button.setEnabled(True)
        self.cancel_training_button.setEnabled(False)
        best_model = str(result.best_model)  # type: ignore[attr-defined]
        self.model_path.setText(best_model)
        self.training_status.append(f"训练完成，最佳模型：{best_model}")
        QMessageBox.information(
            self,
            "训练完成",
            f"最佳模型：\n{best_model}\n\n路径已填入模型框，点击“加载/热切换模型”即可使用。",
        )

    def _on_training_failed(self, worker: ModelTrainingWorker, details: str) -> None:
        if worker is not self.training_worker:
            return
        self.start_training_button.setEnabled(True)
        self.cancel_training_button.setEnabled(False)
        self.training_status.append(details)
        if not details.startswith("用户已取消"):
            QMessageBox.warning(self, "训练失败", details.split("\n", 1)[0])

    def load_model(self) -> None:
        try:
            value = self.model_path.text().strip() or "models/yolov8n.pt"
            self.module.config.detector.confidence = self.confidence.value()
            self.module.config.detector.device = self.device_combo.currentText()
            self.model_status.setText("正在加载模型，请稍候…")
            QApplication.processEvents()
            self.module.load_detector(value)
            resolved = str(self.module.config.detector.model_path)
            self.model_path.setText(resolved)
            save_config(self.module.config)
            names = self.module.detector.class_names
            self.model_status.setText(
                f"模型已加载：{resolved}\n类别：\n"
                + json.dumps(names, ensure_ascii=False, indent=2)
            )
            self._log("YOLO 模型加载成功")
        except Exception as exc:
            self.model_status.setText(str(exc))
            QMessageBox.warning(self, "模型加载失败", str(exc))

    def _refresh_configuration(self) -> None:
        try:
            payload = {
                "health": self.module.health(),
                "camera": vars(self.module.config.camera),
                "depth": vars(self.module.config.depth),
                "detector": vars(self.module.config.detector),
                "pose": vars(self.module.config.pose),
                "board": vars(self.module.config.calibration.board),
                "public_api": [
                    "list_devices/start/stop/read_frame",
                    "list_color_profiles/list_depth_profiles/list_stream_profiles",
                    "get_stream_config/configure_camera",
                    "get_controls/set_control",
                    "detect/estimate_objects/process_once",
                    "point_cloud/capture_snapshot/export_factory_intrinsics",
                    "depth_at/deproject/save_point_cloud_ply/latest_imu",
                    "new_intrinsic_session/new_handeye_session/new_eye_to_hand_session",
                    "set_robot_pose_provider/set_tool_T_camera/set_base_T_camera",
                    "YoloTrainingService.validate/train",
                ],
            }
            self.health_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        except Exception as exc:
            self.health_text.setPlainText(str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            QMessageBox.information(
                self,
                "训练正在停止",
                "已请求取消训练。训练会在当前批次结束后停止，请稍候再关闭窗口。",
            )
            event.ignore()
            return
        self.stop_stream(wait=True)
        self.module.close()
        event.accept()
