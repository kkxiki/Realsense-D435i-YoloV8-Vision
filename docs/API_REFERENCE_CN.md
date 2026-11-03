# D435i 相机模块接口说明

本文档是机械臂接入人员使用的接口手册。代码目录：

```text
/home/user/d435i_camera_module
D:\XuK-许锴\XuK\d435i_camera_module
```

不要修改 `0814/0815` 或其他机械臂工程。相机模块只负责相机、RGB-D、YOLO、物体三维
信息和标定，不负责控制机械臂运动。

## 1. 接口位置速查

| 功能 | 代码位置 | 对外入口 |
|---|---|---|
| Python主接口 | `src/d435i_camera/service.py` | `d435i_camera.CameraModule` |
| 数据模型 | `src/d435i_camera/models.py` | `FrameBundle`、`Detection2D`、`Pose6D`、`PerceptionFrame` |
| 相机参数 | `src/d435i_camera/config.py` | `AppConfig`、`CameraConfig` |
| OpenCV内参标定 | `src/d435i_camera/calibration/intrinsic.py` | `IntrinsicCalibrationSession` |
| 末端相机眼在手上 | `src/d435i_camera/calibration/handeye.py` | `HandEyeCalibrationSession` |
| 头部固定相机眼在手外 | `src/d435i_camera/calibration/eye_to_hand.py` | `EyeToHandCalibrationSession` |
| YOLO框标注 | `src/d435i_camera/annotation.py` | `YoloAnnotationProject`、`YoloBox` |
| YOLO训练 | `src/d435i_camera/training.py` | `YoloTrainingService`、`d435i-yolo-train` |
| 图形界面 | `src/d435i_camera/gui/main_window.py` | `d435i-camera-gui` |
| ROS2 Foxy消息 | `ros2_ws/src/xuk_d435i_interfaces` | 自定义消息和服务 |
| ROS2 Foxy桥接 | `ros2_ws/src/xuk_d435i_bridge` | `xuk_d435i_bridge` |

Python接口和GUI可以不使用ROS2。ROS2接口需要同时启动gateway和bridge，因为相机/YOLO
使用Python 3.10，而ROS2 Foxy的`rclpy`使用系统Python ABI。

## 2. Python主接口

### 2.1 初始化、启动和停止

```python
from d435i_camera import CameraModule, load_config

config = load_config("config/default.yaml")
module = CameraModule(config)
device = module.start()          # CameraDeviceInfo
result = module.process_once()   # RGB-D + YOLO + 6D + 叠加图
module.stop()
module.close()
```

状态属性：

```python
module.running       # bool，相机是否运行
module.latest        # Optional[PerceptionFrame]，最近完整结果
module.latest_raw    # Optional[FrameBundle]，最近原始RGB-D帧
module.latest_imu    # Optional[ImuSample]
module.tool_T_camera # 4x4 ndarray，末端相机外参
module.base_T_camera # 4x4 ndarray，头部固定相机外参
```

### 2.2 设备、分辨率、帧率和硬件参数

```python
devices = module.list_devices()
all_profiles = module.list_stream_profiles()
color_profiles = module.list_color_profiles()
depth_profiles = module.list_depth_profiles()
current = module.get_stream_config()
```

RGB和Depth可以独立设置；旧参数`width/height/fps`仍表示同时设置两路：

```python
module.configure_camera(
    color_width=1280, color_height=720, color_fps=30,
    depth_width=848, depth_height=480, depth_fps=30,
    enable_imu=True,
    restart_if_running=True,
    persist=True,                 # 保存到config/default.yaml
)
```

常用参数还包括`controls={"color.exposure": 100, "depth.laser_power": 150}`、
`validate_streams=True`。正在运行时修改流会自动重启；不支持的组合会回滚。

```python
controls = module.get_controls()
module.set_control("color.brightness", 5)
module.set_control("depth.laser_power", 150)
module.set_controls({"color.exposure": 100, "color.gain": 16})
```

### 2.3 RGB-D、YOLO和6D结果

```python
frame = module.read_frame(timeout_ms=1000)
detections, inference_ms = module.detect(frame)
observations, transform = module.estimate_objects(frame, detections)
result = module.process_once()
```

`PerceptionFrame`主要字段：`frame`（`FrameBundle`）、`observations`（物体列表）、
`annotated_bgr`（叠加图）、`processing_ms`、`inference_ms`和`sequence`。

每个`ObjectObservation`包含：

```python
item.detection.class_id
item.detection.class_name
item.detection.confidence        # 0~1
item.detection.bbox_xyxy         # (x1, y1, x2, y2)，像素
item.pose                         # Optional[Pose6D]
item.valid                        # 是否可用于下游抓取
item.message                      # 无效原因或质量提示
item.depth_m                      # 框内稳健深度，米
```

`Pose6D`字段：

```python
pose.translation_m   # (x, y, z)，米
pose.rpy_deg         # (rx, ry, rz)，度
pose.quaternion_xyzw # 四元数
pose.dimensions_m    # 三维框尺寸，米
pose.frame_id        # camera_color_optical_frame 或 base_link
pose.method           # depth_pca 等
pose.quality          # 姿态质量指标
pose.matrix           # 4x4变换矩阵
```

默认坐标系是`camera_color_optical_frame`。只有加载手眼矩阵并提供有效机械臂位姿后，才
会输出`base_link`下的结果。普通YOLO模型只预测二维框；三维框、XYZ、Rx/Ry/Rz由D435i
深度点云估计，不是CAD级精确6D姿态。

### 2.4 深度、点云和文件输出

```python
depth = module.depth_at(424, 240)             # Optional[float]，米
xyz = module.deproject(424, 240)              # ndarray [x,y,z]，米
points, rgb = module.point_cloud(stride=4)    # Nx3米制点云和Nx3 RGB
module.save_point_cloud_ply("data/cloud.ply", stride=4)
paths = module.capture_snapshot("data/captures", prefix="bottle")
```

快照包括`*_color.png`、`*_depth_m.npy`（有深度时）和`*_metadata.json`。训练标注使用彩色
图，不要把深度文件或元数据当作训练图片。

### 2.5 模型和健康检查

```python
module.load_detector("models/yolov8n.pt")
module.unload_detector()
health = module.health()
```

支持`.pt`、`.onnx`、`.engine`；训练基础模型必须是`.pt`。`health()`返回相机状态、模型
路径、模型是否就绪、手眼矩阵状态、最近帧号、耗时和物体数。

## 3. 标定接口

当前棋盘参数为：内角点`8 × 11`，方格边长`15 mm`。每种标定都先创建会话，重复采样，
最后求解并保存结果。GUI“标定中心”调用的就是这些接口。

### 3.1 OpenCV内参标定

```python
session = module.new_intrinsic_session()
session.reset()
ok, message, preview = session.add_frame(frame.color_bgr)
result = session.solve()                 # 至少12个有效、不同视角
path = result.save("data/calibration/intrinsics.json")
```

`add_frame()`自动检测棋盘、拒绝重复视角并返回预览图；结果包含内参和总体/逐帧重投影
误差。

### 3.2 末端相机：眼在手上

相机固定在机械臂末端、棋盘固定在基座。机械臂侧每次提供同步的`base_T_tool`（4×4，
平移单位米）：

```python
session = module.new_handeye_session()
ok, message, preview = session.add_frame(
    frame.color_bgr, base_T_tool, frame.timestamp_s
)
result = session.solve()
path = result.save("data/calibration/tool_T_camera.json")
module.set_tool_T_camera(result.tool_T_camera)
```

结果主矩阵是`tool_T_camera`，表示“相机坐标转换到末端坐标”。运行时还要通过
`module.set_robot_pose_provider(provider)`提供最新`base_T_tool`。

### 3.3 头部固定相机：眼在手外

相机固定在头部/基座、棋盘刚性固定在机械臂末端：

```python
session = module.new_eye_to_hand_session()
ok, message, preview = session.add_frame(
    frame.color_bgr, base_T_tool, frame.timestamp_s
)
result = session.solve()
path = result.save("data/calibration/base_T_camera.json")
module.set_base_T_camera(result.base_T_camera)
```

结果主矩阵是`base_T_camera`，满足：

```text
base_T_tool × tool_T_board = base_T_camera × camera_T_board
```

姿态变化不足、棋盘未检测到或样本不足时，`add_frame()`返回`False`，不要把该样本用于求解。
## 4. 数据标注和训练接口

### 4.1 可视化标注

GUI路径：“YOLO模型” → “可视化标注图片”。用鼠标拖出框，程序自动计算中心点和宽高，
生成标准YOLO标签，不需要人工计算。

Python侧也可以使用：

```python
from d435i_camera import YoloAnnotationProject, YoloBox

project = YoloAnnotationProject("datasets/custom/data.yaml")
images = project.images("train")
box = YoloBox.from_corners(class_id=0, x1=0.20, y1=0.15, x2=0.70, y2=0.90)
project.save_boxes(images[0], [box])
```

`YoloBox`自动转换为`class_id x_center y_center width height`，四个坐标归一化到`0~1`。
`project.import_images(paths, "train")`可导入图片，`project.import_bgr(image, "val")`可
保存一帧BGR图，`project.set_class_names(["水瓶", "杯子"])`可更新类别。

### 4.2 校验和训练

```python
from d435i_camera import TrainingConfig, YoloTrainingService

request = TrainingConfig(
    dataset_yaml="datasets/custom/data.yaml",
    base_model="models/yolov8n.pt",
    task="detect",
    epochs=100, image_size=640, batch=8,
    device="cpu",                 # NVIDIA GPU填写 "0"
    name="custom_object",
)
service = YoloTrainingService("/home/user/d435i_camera_module")
summary = service.validate(request)
result = service.train(request, progress=print)
print(result.best_model)
```

命令行：

```bash
cd /home/user/d435i_camera_module
source .venv/bin/activate
d435i-yolo-train --data datasets/custom/data.yaml --validate-only
d435i-yolo-train --data datasets/custom/data.yaml --model models/yolov8n.pt --task detect --epochs 100 --imgsz 640 --batch 8 --device cpu
```

训练结果位于`training_runs/<任务名>/weights/best.pt`。训练接口只处理识别模型；三维框和
6D仍使用D435i深度估计。

## 5. ROS2 Foxy接口

### 5.1 启动

终端1启动Python 3.10网关：

```bash
cd /home/user/d435i_camera_module
source .venv/bin/activate
d435i-camera-gateway --config config/default.yaml
```

终端2启动ROS2 Foxy bridge：

```bash
source /opt/ros/foxy/setup.bash
source /home/user/d435i_camera_module/ros2_build_ws/install/setup.bash
ros2 launch xuk_d435i_bridge d435i_bridge.launch.py
```

GUI、gateway不能同时占用D435i；使用ROS2时先停止GUI。两进程只通过本机
`127.0.0.1:17630/17631`通信。

### 5.2 话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/d435i/color/image_raw` | `sensor_msgs/msg/Image` | `bgr8`彩色图 |
| `/d435i/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | `32FC1`，米制对齐深度 |
| `/d435i/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 当前RGB内参 |
| `/d435i/annotated/image_raw` | `sensor_msgs/msg/Image` | 3D框、名称、置信度和6D叠加图 |
| `/d435i/objects_6d` | `xuk_d435i_interfaces/msg/Object6DArray` | 推荐给抓取节点 |
| `/yolov8/detections` | `vision_msgs/msg/Detection2DArray` | 兼容二维检测 |

推荐订阅`/d435i/objects_6d`，不要从叠加图OCR姿态信息。`Object6D`字段：

```text
class_name / class_name_zh       英文/中文类别
class_id / confidence             类别编号/置信度
bbox_xyxy_px                     像素框 [x1,y1,x2,y2]
pose.position                    XYZ，米
pose.orientation                 四元数 xyzw
dimensions_m                     三维框尺寸，米
rx_deg / ry_deg / rz_deg         RPY，度
frame_id                         坐标系
valid / message                  是否可用及原因
depth_m                          框内深度，米
```

### 5.3 服务

| 服务 | 类型 | 说明 |
|---|---|---|
| `/d435i/start` | `std_srvs/srv/Trigger` | 启动相机 |
| `/d435i/stop` | `std_srvs/srv/Trigger` | 停止相机 |
| `/d435i/snapshot` | `std_srvs/srv/Trigger` | 保存RGB-D和JSON |
| `/d435i/get_json` | `xuk_d435i_interfaces/srv/GetJson` | `status`、`profiles`、`controls` |
| `/d435i/set_stream` | `xuk_d435i_interfaces/srv/SetStream` | 独立设置RGB/Depth分辨率帧率 |
| `/d435i/set_control` | `xuk_d435i_interfaces/srv/SetControl` | 设置RealSense硬件参数 |
| `/d435i/calibration` | `xuk_d435i_interfaces/srv/CalibrationCommand` | `reset/capture/solve`标定 |

示例：

```bash
ros2 service call /d435i/get_json xuk_d435i_interfaces/srv/GetJson \
  "{operation: status}"

ros2 service call /d435i/set_stream xuk_d435i_interfaces/srv/SetStream \
  "{color_width: 848, color_height: 480, color_fps: 30, depth_width: 848, depth_height: 480, depth_fps: 30, enable_imu: true, persist: true}"

ros2 service call /d435i/set_control xuk_d435i_interfaces/srv/SetControl \
  "{name: color.exposure, value: 100.0}"
```

标定服务的`calibration_type`为`intrinsic`、`eye_in_hand`或`eye_to_hand`，`command`为
`reset`、`capture`或`solve`。眼在手上/眼在手外的`base_to_tool`必须由机械臂节点在每次
`capture`时填写；相机接口不会发送运动指令。

## 6. 接入时必须遵守的约定

1. 长度统一使用米；角度统一使用度；图像坐标使用像素。
2. 抓取前检查`Object6D.valid == true`，并确认`frame_id`是抓取节点需要的坐标系。
3. 相机未完成手眼标定或机器人位姿过期时，结果会保留相机坐标并标记无效原因，不能直接
   当作`base_link`坐标使用。
4. `dimensions_m`和`depth_m`是深度估计值，不是物品数据库中的理论尺寸。
5. `d435i-camera-gui`、`d435i-camera-gateway`和ROS bridge只能选一个进程组占用相机。
6. 机械臂适配只需要实现`RobotPoseProvider.latest()`，返回最新的`base_T_tool` 4×4齐次
   矩阵；具体约定见`docs/ROBOT_POSE_ADAPTER.md`。

## 7. 目录结构和系统兼容性

### 7.1 目录结构

```text
d435i_camera_module/
├── config/                 配置文件，默认是config/default.yaml
├── data/                   标定结果、快照和运行数据
├── datasets/               YOLO数据集，包含images/、labels/和data.yaml
├── docs/                   中文接口、ROS2、标定、训练说明
├── models/                 YOLO基础模型和训练后的best.pt
├── ros2_ws/                ROS2接口源码
│   └── src/
│       ├── xuk_d435i_interfaces/  Object6D消息和控制/标定服务
│       └── xuk_d435i_bridge/       ROS2节点和launch文件
├── scripts/                安装、启动、下载模型和训练冒烟脚本
├── src/d435i_camera/       Python主程序源码
│   ├── camera/             RealSense和Mock相机后端
│   ├── calibration/        OpenCV、眼在手上、眼在手外标定
│   └── gui/                PyQt5图形界面和可视化标注工具
├── tests/                  单元测试
├── pyproject.toml          Python依赖、命令行入口和安装配置
└── README_CN.md            项目总说明
```

`models/`、`data/`和`datasets/`是运行数据目录；源代码主要在`src/`和`ros2_ws/src/`。
`.venv/`、`ros2_build_ws/`和缓存目录是某台机器生成的环境/构建产物，不能直接复制到另一台
系统使用，应在目标系统重新创建。

### 7.2 已验证环境和迁移规则

| 系统 | Python/ROS2 | 结论 |
|---|---|---|
| Ubuntu 20.04 | Python 3.10虚拟环境 + ROS2 Foxy系统Python | 当前已验证，推荐环境 |
| Ubuntu 22.04 | Python 3.10–3.12；ROS2 Humble | Python核心和GUI通常可迁移；ROS2包需在Humble工作区重新编译和测试 |
| Ubuntu 24.04 | Python 3.10–3.12；ROS2 Jazzy | 核心代码可尝试迁移；需要重新安装RealSense/Ultralytics并重新编译ROS2包，未作为当前交付环境验证 |
| Windows 10/11 | Python 3.10–3.12 | Python核心、Mock模式和GUI可迁移；RealSense需安装Windows版SDK/USB驱动；当前ROS2 Foxy桥接流程不是Windows交付目标 |
| Debian/其他Linux | Python 3.10–3.12 | 核心代码理论可运行，前提是安装librealsense、pyrealsense2、OpenCV和Qt；需要现场验证USB权限和设备规则 |
| macOS | — | 不作为当前支持目标，RealSense和ROS2硬件链路需要单独适配 |

迁移时不要复制Ubuntu上的`.venv`。在目标系统执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all]"
```

然后按目标系统安装`librealsense`和对应的`pyrealsense2`。只测试算法或界面时，可把
`config/default.yaml`中的`camera.backend`设为`mock`，这样不需要连接D435i。

ROS2迁移必须在目标ROS发行版中重新执行`colcon build --symlink-install`，不能复用
Ubuntu 20.04/Foxy生成的`ros2_build_ws/install`。若ROS2发行版改变，优先保留接口消息和
话题名称，再根据新发行版重新编译和验证bridge。
