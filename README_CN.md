# XuK D435i 相机与 6D 视觉模块（Ubuntu）

本项目面向 Ubuntu 20.04 桌面环境。相机为 Intel RealSense D435i，
同时支持末端相机“眼在手上（eye-in-hand）”和头部固定相机“眼在手外（eye-to-hand）”。
Windows 目录仅作为源码交付备份，不是运行环境。

## 已实现功能

- D435i 枚举、启动、停止和健康状态；可按序列号锁定指定设备（`camera.serial`，留空则自动选择第一台）。
- 分别查询 RGB 彩色 BGR8 与 Depth 深度 Z16 的原生 profile，并独立修改两路分辨率和
  帧率；组合会在保存前实机启动验证，不兼容时自动回滚。旧的共同 profile API 保留兼容。
- 彩色图、对齐到彩色坐标的米制深度、相机内参、像素深度/去投影、PLY 点云和可选 IMU 数据接口。
- 动态枚举 44 项实机选项，以 `color.* / depth.* / motion.*` 区分同名参数；RGB 亮度、
  对比度、Gamma、白平衡等与 Depth 曝光、增益、发射器、激光、预设等可独立调整。
  会改变深度单位、同步或固件时序的高级项可查看但默认禁止修改。
- 彩色/深度/元数据快照；元数据包含检测与 6D 结果。
- Ultralytics YOLOv8 `.pt/.onnx/.engine` 热加载、CPU 推理、类别与阈值过滤。
- 检测模型用框内深度点，分割模型优先用实例 mask 内深度点。
- 中值深度去投影、点云 PCA 通用物体姿态、时序平滑、质量评分。
- 腕部相机坐标链：`base_T_tool × tool_T_camera × camera_T_object`。
- 头部固定相机坐标链：`base_T_camera × camera_T_object`。
- OpenCV 棋盘内参标定、RealSense 出厂内参导出。
- 眼在手上手眼采集与 PARK/TSAI/HORAUD/ANDREFF/DANIILIDIS 求解及残差报告。
- 头部固定相机眼在手外采集，以 OpenCV SHAH/LI 联合求解 `base_T_camera` 和
  `tool_T_board`，并报告方程残差。
- PyQt5 GUI：实时画面、物体三维框旁的“物品中文名称 + 置信度 + X/Y/Z/Rx/Ry/Rz”、
  坐标轴、坐标系切换、设备参数、三种标定流程和模型入口。
- 无相机/无模型也可运行的 RGB-D 模拟模式和自动化测试。

## Ubuntu 安装

主机当前已有 librealsense 2.56.5、OpenCV、PyQt5、Ultralytics 和 ROS Foxy/Noetic。
默认 Python 是 3.10，但缺少 `pyrealsense2`，安装脚本会创建复用系统包的虚拟环境并补齐绑定：

```bash
cd /home/user/d435i_camera_module
chmod +x scripts/*.sh
bash scripts/install.sh
```

真实 D435i GUI：

```bash
scripts/run_gui.sh
```

离线模拟 GUI：

```bash
scripts/run_mock_gui.sh
```

命令行自检：

```bash
source .venv/bin/activate
d435i-camera --config config/default.yaml info
d435i-camera --config config/default.yaml --mock run --frames 10
pytest
```

## 开源预训练模型

当前已经下载并接入 `models/yolov8n.pt`，默认启用全部 80 个 COCO 类别，包括人、水瓶、
杯子、手机等，GUI 使用完整中文类别名称，默认置信度为 0.30。需要限制类别时可在
`config/default.yaml` 的 `detector.classes` 填写类别 ID。后续需要更干净的深度点时可改用：

- `yolov8n.pt`：普通检测，CPU 最快。
- `yolov8n-seg.pt`：实例分割，物体深度点更干净，通用姿态通常更稳。

官方 YOLOv8 COCO 模型包含 80 个常见类别，例如 person、bottle、cup、bowl、banana、
apple、chair 等。下载必须显式确认许可证：

```bash
source .venv/bin/activate
python scripts/download_model.py yolov8n-seg.pt --accept-agpl
```

下载其他模型后在 `config/default.yaml` 设置：

```yaml
detector:
  model_path: models/yolov8n-seg.pt
```

Ultralytics 代码和训练模型默认是 AGPL-3.0。若该机器人项目是闭源、内部专有或商业部署，
请在采用前由项目负责人确认 AGPL 合规或取得 Ultralytics 商业许可证。检测器已抽象为
`ObjectDetector`，需要时可换成其他许可/推理后端而不修改相机、标定和 GUI。

## 标定参数与流程

用户确认的棋盘为：`8 × 11` 内角点、单格 `15 mm`。配置已写入并标记 confirmed。

内参标定：

1. 启动真实相机，进入“标定中心 → OpenCV 内参标定”。
2. 棋盘覆盖画面中央、四角、远近和不同倾角；每次清晰静止后采集。
3. 重复视角会自动拒绝，至少 12 个有效视角。
4. 求解并检查总体/逐帧重投影误差，再保存 `data/calibration/intrinsics.json`。

眼在手上手眼标定：

1. 棋盘刚性固定在机器人基座，D435i 刚性固定在末端。
2. 标定期间每次机械臂完全停止，确保相机帧与 `base_T_tool` 同步后采集。
3. 至少 12 个、建议 15–25 个明显不同的位置和朝向。
4. 求解得到 `tool_T_camera`；报告中的平移/旋转残差必须经独立实测验证。
5. 标定程序不会向机械臂发送运动指令。

头部固定相机眼在手外标定：

1. D435i 固定在机器人头部/基座，棋盘刚性固定在末端工具上。
2. 每次机械臂静止后同步采集图像和 `base_T_tool`，至少 12 个、建议 15–25 个多样姿态。
3. OpenCV 联合求解 `base_T_camera` 与 `tool_T_board`，满足
   `base_T_tool × tool_T_board = base_T_camera × camera_T_board`。
4. 结果保存到 `data/calibration/base_T_camera.json`；运行时在 GUI 选择
   “基座坐标系（头部固定相机）”。
5. 如果“头部”是会运动的关节，标定期间必须保持头部不动；运行时由机械臂侧提供
   `base_T_head`，将固定外参扩展为 `base_T_head × head_T_camera`。

## 6D 定义与限制

实时画面不显示左上角汇总面板；每个物体三维框旁显示“物品中文名称、置信度、
X/Y/Z（mm）、Rx/Ry/Rz（度）”。质量、帧号和耗时不会混入标签。Rx/Ry/Rz 使用固定
XYZ/RPY 表示。未接机械臂
位姿时输出 `camera_color_optical_frame`；加载相应手眼矩阵后可在 GUI 切换到 `base_link`。

普通检测/分割模型不直接预测类别级 6D。本模块的 `depth_pca` 是无 CAD 的通用几何姿态：
适合瓶、盒、长条等主轴明显物体；对球体、圆柱绕轴、对称件、遮挡/堆叠物体存在不可观测
或 180° 歧义。GUI 会显示方法和质量，点数不足时标记无效。要求精确物体姿态时，应为相应
类别增加 CAD/关键点/6D Pose 插件；`ObjectPoseEstimator` 接口允许扩展。

## 机械臂适配

相机模块只读机械臂末端位姿，不控制机械臂。机械臂开发人员实现
`RobotPoseProvider.latest()` 并返回米制 `base_T_tool` 即可。具体约定见
`docs/ROBOT_POSE_ADAPTER.md`。在 ROS topic、消息类型和字段确定前，GUI 保留手动位姿入口，
不会猜测消息定义。

## 主要 Python API

```python
from d435i_camera import CameraModule, load_config

module = CameraModule(load_config("config/default.yaml"))
device = module.start()
result = module.process_once()           # 完整 RGB-D + YOLO + 6D + 叠加
depth = module.depth_at(424, 240)        # ROI 中值深度，单位 m
xyz = module.deproject(424, 240)         # 光学坐标系 XYZ，单位 m
points, rgb = module.point_cloud()       # Nx3 米制点云
module.save_point_cloud_ply("cloud.ply")
paths = module.capture_snapshot()        # 彩色、深度、JSON
controls = module.get_controls()          # color.* / depth.* / motion.*
module.set_control("color.brightness", 5)
module.set_control("depth.laser_power", 150)
color_profiles = module.list_color_profiles()
depth_profiles = module.list_depth_profiles()
module.configure_camera(
    color_width=1280, color_height=720, color_fps=30,
    depth_width=848, depth_height=480, depth_fps=30,
    enable_imu=True,
    restart_if_running=True, persist=True,
)
module.close()
```

完整接口还包括 `list_devices/list_color_profiles/list_depth_profiles/list_stream_profiles/`
`get_stream_config/configure_camera/`
`read_frame/detect/estimate_objects/export_factory_intrinsics/`
`new_intrinsic_session/new_handeye_session/new_eye_to_hand_session/`
`set_robot_pose_provider/set_tool_T_camera/set_base_T_camera`。

接口位置、参数、返回值、标定、标注训练和 ROS2 Foxy 话题/服务的完整说明见
[`docs/API_REFERENCE_CN.md`](docs/API_REFERENCE_CN.md)。

## ROS2 Foxy 接口

独立 ROS2 包位于 `ros2_ws/src/xuk_d435i_interfaces` 和
`ros2_ws/src/xuk_d435i_bridge`。它们只发布相机能力，不接入或修改任何现有机械臂、抓取
工作空间。完整的话题、服务、消息字段、构建和调用示例见
[`docs/ROS2_FOXY_API.md`](docs/ROS2_FOXY_API.md)。

## 自定义物品训练接口

提供鼠标画框的可视化标注工具、`YoloAnnotationProject` 数据集API、
`YoloTrainingService` 训练API、`d435i-yolo-train` 命令行和 GUI 后台训练入口。
在GUI中框住物体后会自动计算并保存YOLO中心点和宽高，不需要人工填写坐标。
训练前会校验图像、标签、类别编号和归一化坐标，训练完成返回 `best.pt`。数据目录模板和
完整步骤见 [`docs/MODEL_TRAINING.md`](docs/MODEL_TRAINING.md)。
