"""Mouse-driven YOLO detection annotation dialog."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..annotation import YoloAnnotationProject, YoloBox
from ..training import IMAGE_SUFFIXES


COLORS = [
    QColor("#4cc9f0"),
    QColor("#f72585"),
    QColor("#80ed99"),
    QColor("#ffb703"),
    QColor("#b388ff"),
    QColor("#fb8500"),
]


class AnnotationCanvas(QWidget):
    box_created = pyqtSignal(float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._boxes: List[YoloBox] = []
        self._names: Dict[int, str] = {}
        self._drag_start: Optional[QPoint] = None
        self._drag_end: Optional[QPoint] = None

    def set_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._drag_start = None
        self._drag_end = None
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self._boxes = []
        self.update()

    def set_boxes(self, boxes: List[YoloBox], names: Dict[int, str]) -> None:
        self._boxes = list(boxes)
        self._names = dict(names)
        self.update()

    def _image_rect(self) -> Optional[QRectF]:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        width = self._pixmap.width() * scale
        height = self._pixmap.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _normalized(self, point: QPoint) -> Optional[tuple[float, float]]:
        rect = self._image_rect()
        if rect is None or not rect.contains(point):
            return None
        return ((point.x() - rect.left()) / rect.width(), (point.y() - rect.top()) / rect.height())

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080b0f"))
        rect = self._image_rect()
        if rect is None or self._pixmap is None:
            painter.setPen(QColor("#8793a3"))
            painter.drawText(self.rect(), Qt.AlignCenter, "请导入图片或加入当前相机帧")
            return
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setFont(QFont("Sans", 10, QFont.Bold))
        for box in self._boxes:
            x1, y1, x2, y2 = box.corners()
            box_rect = QRectF(
                rect.left() + x1 * rect.width(),
                rect.top() + y1 * rect.height(),
                (x2 - x1) * rect.width(),
                (y2 - y1) * rect.height(),
            )
            color = COLORS[box.class_id % len(COLORS)]
            painter.setPen(QPen(color, 3))
            painter.drawRect(box_rect)
            label = f"{box.class_id}: {self._names.get(box.class_id, str(box.class_id))}"
            text_width = painter.fontMetrics().horizontalAdvance(label) + 10
            text_rect = QRectF(box_rect.left(), max(rect.top(), box_rect.top() - 23), text_width, 23)
            painter.fillRect(text_rect, color)
            painter.setPen(QColor("#071018"))
            painter.drawText(text_rect.adjusted(5, 0, -2, 0), Qt.AlignVCenter, label)
        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.DashLine))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._normalized(event.pos()) is not None:
            self._drag_start = event.pos()
            self._drag_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None:
            return
        rect = self._image_rect()
        if rect is None:
            return
        self._drag_end = QPoint(
            int(max(rect.left(), min(rect.right(), event.x()))),
            int(max(rect.top(), min(rect.bottom(), event.y()))),
        )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._drag_start is None or self._drag_end is None:
            return
        start = self._normalized(self._drag_start)
        end = self._normalized(self._drag_end)
        drag = QRectF(self._drag_start, self._drag_end).normalized()
        self._drag_start = None
        self._drag_end = None
        self.update()
        if start is not None and end is not None and drag.width() >= 4 and drag.height() >= 4:
            self.box_created.emit(start[0], start[1], end[0], end[1])


class AnnotationDialog(QDialog):
    """Operator UI that converts mouse rectangles into YOLO labels automatically."""

    def __init__(
        self,
        project: YoloAnnotationProject,
        current_frame: Optional[Callable[[], Optional[np.ndarray]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.current_frame = current_frame
        self.current_image: Optional[Path] = None
        self.boxes: List[YoloBox] = []
        self.setWindowTitle("YOLO 可视化数据标注")
        self.resize(1320, 820)
        self.setModal(False)
        self._build_ui()
        self._refresh_classes()
        self._refresh_images(select_index=0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"数据集：{self.project.yaml_path}"))
        instructions = QLabel(
            "操作：选择类别 → 在图片上按住左键拖框 → 松开即自动保存。中心点和宽高由程序自动计算。"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color:#72a7ff; font-weight:bold;")
        root.addWidget(instructions)

        toolbar = QHBoxLayout()
        self.split_combo = QComboBox()
        self.split_combo.addItem("训练集 train", "train")
        self.split_combo.addItem("验证集 val", "val")
        self.split_combo.currentIndexChanged.connect(lambda _index: self._refresh_images(0))
        import_button = QPushButton("导入图片…")
        import_button.clicked.connect(self._import_images)
        camera_button = QPushButton("加入当前相机帧")
        camera_button.clicked.connect(self._import_camera_frame)
        edit_classes = QPushButton("管理类别…")
        edit_classes.clicked.connect(self._edit_classes)
        previous = QPushButton("上一张")
        previous.clicked.connect(lambda: self._move(-1))
        following = QPushButton("下一张")
        following.clicked.connect(lambda: self._move(1))
        toolbar.addWidget(QLabel("当前分组"))
        toolbar.addWidget(self.split_combo)
        toolbar.addWidget(import_button)
        toolbar.addWidget(camera_button)
        toolbar.addWidget(edit_classes)
        toolbar.addStretch(1)
        toolbar.addWidget(previous)
        toolbar.addWidget(following)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(220)
        self.image_list.currentRowChanged.connect(self._load_row)
        splitter.addWidget(self.image_list)
        self.canvas = AnnotationCanvas()
        self.canvas.box_created.connect(self._add_box)
        splitter.addWidget(self.canvas)

        panel = QWidget()
        panel.setMinimumWidth(275)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(QLabel("画框使用的物品类别"))
        self.class_combo = QComboBox()
        panel_layout.addWidget(self.class_combo)
        panel_layout.addWidget(QLabel("当前图片中的标注框"))
        self.box_list = QListWidget()
        panel_layout.addWidget(self.box_list, 1)
        delete_button = QPushButton("删除选中的框")
        delete_button.setObjectName("danger")
        delete_button.clicked.connect(self._delete_box)
        panel_layout.addWidget(delete_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        panel_layout.addWidget(self.status)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        root.addWidget(splitter, 1)

    @property
    def split(self) -> str:
        return str(self.split_combo.currentData())

    def _refresh_classes(self) -> None:
        self.class_combo.clear()
        for class_id, name in self.project.class_names.items():
            self.class_combo.addItem(f"{class_id}: {name}", class_id)

    def _refresh_images(self, select_index: int = 0, select_path: Optional[Path] = None) -> None:
        images = self.project.images(self.split)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in images:
            self.image_list.addItem(path.name)
            self.image_list.item(self.image_list.count() - 1).setData(Qt.UserRole, str(path))
        self.image_list.blockSignals(False)
        if select_path is not None:
            matching = [index for index, path in enumerate(images) if path == select_path]
            select_index = matching[0] if matching else select_index
        if images:
            self.image_list.setCurrentRow(max(0, min(select_index, len(images) - 1)))
            self._load_row(self.image_list.currentRow())
        else:
            self.current_image = None
            self.boxes = []
            self.canvas.clear_image()
            self.box_list.clear()
            self.status.setText(f"{self.split} 中还没有图片")

    def _load_row(self, row: int) -> None:
        item = self.image_list.item(row)
        if item is None:
            return
        path = Path(str(item.data(Qt.UserRole)))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片：{path}")
            return
        try:
            self.boxes = self.project.read_boxes(path)
        except Exception as exc:
            QMessageBox.warning(self, "标签错误", str(exc))
            self.boxes = []
        self.current_image = path
        self.canvas.set_image(image)
        self._refresh_boxes()
        self.status.setText(
            f"{self.split}：第 {row + 1}/{self.image_list.count()} 张；"
            f"当前 {len(self.boxes)} 个框（允许0个框作为负样本）"
        )

    def _refresh_boxes(self) -> None:
        self.box_list.clear()
        for index, box in enumerate(self.boxes):
            name = self.project.class_names.get(box.class_id, str(box.class_id))
            self.box_list.addItem(
                f"#{index + 1} {name}  中心({box.x_center:.3f}, {box.y_center:.3f}) "
                f"宽高({box.width:.3f}, {box.height:.3f})"
            )
        self.canvas.set_boxes(self.boxes, self.project.class_names)

    def _save(self) -> None:
        if self.current_image is not None:
            self.project.save_boxes(self.current_image, self.boxes)

    def _add_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.current_image is None or self.class_combo.currentData() is None:
            return
        try:
            self.boxes.append(YoloBox.from_corners(int(self.class_combo.currentData()), x1, y1, x2, y2))
            self._save()
            self._refresh_boxes()
            self.status.setText(f"已自动保存：{_label_path_text(self.current_image)}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def _delete_box(self) -> None:
        row = self.box_list.currentRow()
        if row < 0 or row >= len(self.boxes):
            QMessageBox.information(self, "删除标注", "请先在右侧选择一个标注框")
            return
        del self.boxes[row]
        self._save()
        self._refresh_boxes()

    def _import_images(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_SUFFIXES))
        paths, _ = QFileDialog.getOpenFileNames(self, "选择需要标注的彩色图片", "", f"图片 ({patterns})")
        if not paths:
            return
        imported = self.project.import_images(paths, self.split)
        if imported:
            self._refresh_images(select_path=imported[-1])
            self.status.setText(f"已导入 {len(imported)} 张图片到 {self.split}")

    def _import_camera_frame(self) -> None:
        frame = self.current_frame() if self.current_frame is not None else None
        if frame is None:
            QMessageBox.information(self, "没有相机帧", "请先启动相机，等待实时画面出现后再点击。")
            return
        try:
            path = self.project.import_bgr(frame.copy(), self.split)
            self._refresh_images(select_path=path)
        except Exception as exc:
            QMessageBox.warning(self, "保存相机帧失败", str(exc))

    def _edit_classes(self) -> None:
        existing = "\n".join(self.project.class_names[index] for index in sorted(self.project.class_names))
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "管理物品类别",
            "每行一个类别。已有标签后只能在末尾新增，不能改动已有编号：",
            existing,
        )
        if not accepted:
            return
        try:
            self.project.set_class_names(text.splitlines())
            self._refresh_classes()
            self._refresh_boxes()
        except Exception as exc:
            QMessageBox.warning(self, "类别保存失败", str(exc))

    def _move(self, offset: int) -> None:
        if self.image_list.count() == 0:
            return
        row = max(0, min(self.image_list.currentRow() + offset, self.image_list.count() - 1))
        self.image_list.setCurrentRow(row)


def _label_path_text(image: Path) -> str:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value == "images"]
    if indexes:
        parts[indexes[-1]] = "labels"
    return str(Path(*parts).with_suffix(".txt"))
