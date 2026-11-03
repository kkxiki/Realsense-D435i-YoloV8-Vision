# Realsense D435i · YOLOv8 · 6D 视觉模块

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
<img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-20.04-E95420?logo=ubuntu&logoColor=white">
<img alt="ROS 2" src="https://img.shields.io/badge/ROS%202-Foxy-22314E?logo=ros&logoColor=white">
<img alt="PyQt5" src="https://img.shields.io/badge/GUI-PyQt5-41CD52?logo=qt&logoColor=white">
<img alt="YOLOv8" src="https://img.shields.io/badge/Detection-YOLOv8-00FFFF?logo=yolo&logoColor=black">
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

基于 Intel RealSense D435i 的 RGB-D 视觉模块：相机管理、YOLOv8 目标检测、无 CAD 通用 6D 位姿估计、**双模式手眼标定**、PyQt5 图形界面、自定义物品标注与训练，以及 ROS 2 Foxy 桥接。

设计上分三层：`CameraModule` 提供统一的 Python API，PyQt5 GUI 和 ROS 2 bridge 都只是它的消费者，三者通过本地 TCP（`127.0.0.1:17630/17631`）解耦——GUI 和 bridge 不会同时抢占相机。**相机模块只读机械臂末端位姿，从不向机械臂发送运动指令**。

## 技术指标

| 指标 | 数值 |
| --- | --- |
| 相机 | Intel RealSense D435i，RGB + 深度（对齐到彩色系）+ 可选 IMU |
| 默认分辨率 | 848 × 480 @ 30 fps（彩色与深度可**独立**配置分辨率和帧率） |
| 检测模型 | Ultralytics YOLOv8 `.pt` / `.onnx` / `.engine`，默认 CPU 推理 |
| 开放类别 | COCO 80 类，含完整中文名称；可按 ID 过滤 |
| 位姿估计 | `depth_pca`：点云 PCA 主轴法，无 CAD 依赖 |
| 时序平滑 | `smoothing_alpha = 0.35`，目标跟踪最大中心位移 80 px |
| 实机参数 | 动态枚举 **44 项** D435i 运行时可调参数 |
| 手眼算法 | 眼在手上 5 种：PARK / TSAI / HORAUD / ANDREFF / DANIILIDIS |
| 眼在手外 | OpenCV SHAH / LI 联合求解 `base_T_camera` 与 `tool_T_board` |
| 标定最少样本 | 内参 12 / 手眼 12（建议 15–25），重复视角自动拒绝 |
| 棋盘规格 | 8 × 11 内角点，单格 15 mm |
| 自动化测试 | 31 个测试文件，含无相机、无模型的 RGB-D 模拟后端 |

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        消费者（二选一，互斥）                          │
│   ┌────────────────────────┐        ┌────────────────────────────┐   │
│   │  PyQt5 GUI             │        │  ROS 2 Foxy bridge          │   │
│   │  实时画面 · 6D 框 · 标定 │        │  xuk_d435i_bridge           │   │
│   │  参数面板 · 训练入口     │        │  xuk_d435i_interfaces       │   │
│   └───────────┬────────────┘        └────────────┬───────────────┘   │
└───────────────┼──────────────────────────────────┼───────────────────┘
                │ 本地 TCP 127.0.0.1:17630/17631   │
                ▼                                  ▼
      ┌─────────────────────────────────────────────────────┐
      │                 CameraModule（核心）                  │
      │  ┌───────────┐ ┌──────────┐ ┌─────────────────────┐ │
      │  │ 相机后端   │ │ 检测器    │ │ 位姿估计             │ │
      │  │ realsense │ │ YOLOv8   │ │ depth_pca (PCA)      │ │
      │  │ mock      │ │ onnx     │ │ + 时序平滑 + 质量评分 │ │
      │  └───────────┘ └──────────┘ └─────────────────────┘ │
      │  ┌────────────────────────────────────────────────┐ │
      │  │ 标定：内参(OpenCV) / 眼在手上(5 算法) / 眼在手外 │ │
      │  └────────────────────────────────────────────────┘ │
      │  ┌────────────────────────────────────────────────┐ │
      │  │ 标注 + 训练：YoloAnnotationProject / Training   │ │
      │  └────────────────────────────────────────────────┘ │
      └─────────────────────────────────────────────────────┘
```

## 核心设计

### 1. 双标定模式：眼在手上 + 眼在手外

多数视觉模块只做一种手眼关系，这个模块两种都实现了，且在同一次运行里可切换：

**眼在手上（eye-in-hand）** — 相机装在末端：

```
base_T_object = base_T_tool × tool_T_camera × camera_T_object
```

用 5 种算法分别求解 `tool_T_camera`（PARK / TSAI / HORAUD / ANDREFF / DANIILIDIS），输出各自的残差报告供交叉验证——单一算法的残差好看不代表解对，多算法一致才有说服力。

**眼在手外（eye-to-hand）** — 相机固定在头部/基座：

```
base_T_tool × tool_T_board = base_T_camera × camera_T_board
```

用 OpenCV SHAH/LI **联合求解** `base_T_camera` 与 `tool_T_board` 两个未知量，而不假设棋盘位姿已知。

### 2. 无 CAD 的通用 6D：能力与边界都写清楚

`depth_pca` 用检测框内（分割模型则优先用实例 mask 内）的深度点做去投影，再对点云做 PCA 求主轴，得到任意物体的 6D 位姿。好处是不需要为每个物体建模，瓶子、盒子、长条物都直接能用。

但它有明确的不可观测情形，模块不掩饰、直接标出：

- 球体、圆柱绕轴旋转 → 绕轴自由度不可观测
- 对称件 → 存在 180° 歧义
- 遮挡、堆叠 → 点云不完整，PCA 主轴偏移

GUI 会同时显示所用方法和质量评分，点数不足时标记为无效，而不是硬给一个看起来合理的姿态。需要类别级精确位姿时，`ObjectPoseEstimator` 接口留了扩展点，可接 CAD / 关键点 / 6D Pose 模型。

### 3. 分辨率与帧率的独立配置 + 保存前实机验证

彩色流和深度流分别查询原生 profile，可以各自设不同的分辨率和帧率。关键点是：**组合会在保存前用实机启动验证一次**，不兼容的组合自动回滚到上一个可用配置。这避免了"改完配置重启才发现相机打不开"的常见坑。

### 4. 44 项实机参数的命名空间化

D435i 的彩色流和深度流有很多同名参数（如曝光、增益），直接平铺会互相覆盖。模块把它们按 `color.* / depth.* / motion.*` 分组建模，GUI 里可逐项调整。

会改变深度单位、帧同步或固件时序的高级项**可以查看但默认禁止修改**——这类参数一旦改错，后续所有标定和测量都会失真，而且很难归因。

### 5. 相机抽象层带来可测试性

`camera/base.py` 定义后端接口，`realsense.py` 是实机实现，`mock.py` 是模拟实现。因此 **31 个测试用例在没有任何相机和模型的机器上也能全部跑通**，这是这个模块能持续迭代的基础。

## 目录结构

```
├── config/default.yaml       # 相机、深度、检测、位姿、标定、GUI 全部配置
├── src/d435i_camera/         # 核心 Python 包
│   ├── camera/               # 后端抽象（realsense / mock）
│   ├── calibration/          # 内参 / 手眼 / 眼在手外
│   ├── gui/                  # PyQt5 界面与标注对话框
│   ├── detection.py          # YOLOv8 封装与热加载
│   ├── pose.py               # depth_pca 位姿估计
│   ├── service.py            # CameraModule 门面
│   ├── gateway.py            # 本地 TCP 服务
│   ├── annotation.py         # 标注工程
│   └── training.py           # 训练服务
├── ros2_ws/src/              # ROS 2 Foxy 包
│   ├── xuk_d435i_bridge/     # 桥接节点
│   └── xuk_d435i_interfaces/ # 2 个 msg + 4 个 srv
├── scripts/                  # 安装、启动、模型下载、训练自检
├── tests/                    # 31 个测试
├── datasets/custom/          # YOLO 数据集模板
├── docs/                     # API / ROS2 / 训练 / 机械臂适配 / 测试报告
└── models/                   # 模型目录（权重不入库）
```

## 安装

```bash
cd /home/user/d435i_camera_module
chmod +x scripts/*.sh
bash scripts/install.sh
```

安装脚本会创建复用系统包的虚拟环境并补齐 `pyrealsense2` 绑定（系统默认 Python 是 3.10，往往缺这个绑件）。

## 快速开始

```bash
# 真实 D435i 图形界面
scripts/run_gui.sh

# 无相机离线模拟（用于开发与演示）
scripts/run_mock_gui.sh

# 命令行自检
source .venv/bin/activate
d435i-camera --config config/default.yaml info
d435i-camera --config config/default.yaml --mock run --frames 10
pytest
```

### 模型准备

仓库**不包含** YOLO 权重（默认 `models/yolov8n.pt`）。首次使用需下载：

```bash
source .venv/bin/activate
python scripts/download_model.py yolov8n-seg.pt --accept-agpl
```

> Ultralytics 代码与官方权重默认 **AGPL-3.0**。若本项目用于闭源、内部专有或商业部署，采用前需确认 AGPL 合规或取得 Ultralytics 商业许可证。检测器已抽象为 `ObjectDetector`，可替换为其他许可的推理后端而不影响相机、标定和 GUI。

## 主要 Python API

```python
from d435i_camera import CameraModule, load_config

module = CameraModule(load_config("config/default.yaml"))
device = module.start()
result = module.process_once()            # RGB-D + YOLO + 6D + 叠加
depth  = module.depth_at(424, 240)        # ROI 中值深度（m）
xyz    = module.deproject(424, 240)       # 光学系 XYZ（m）
points, rgb = module.point_cloud()        # Nx3 米制点云
module.save_point_cloud_ply("cloud.ply")
paths  = module.capture_snapshot()        # 彩色 / 深度 / JSON
module.set_control("color.brightness", 5)
module.close()
```

完整接口见 [`docs/API_REFERENCE_CN.md`](docs/API_REFERENCE_CN.md)。

## 标定流程

| 流程 | 最少样本 | 输出 |
| --- | --- | --- |
| OpenCV 内参标定 | 12 个有效视角（重复视角自动拒绝） | `data/calibration/intrinsics.json` |
| 眼在手上手眼标定 | 12（建议 15–25） | `data/calibration/tool_T_camera.json` |
| 眼在手外标定 | 12（建议 15–25） | `data/calibration/base_T_camera.json` |

标定期间每次采集都要求机械臂**完全静止**，保证相机帧与 `base_T_tool` 同步。求解后必须检查总体与逐帧重投影误差，摊在报告里的残差需经独立实测验证。

## ROS 2 Foxy 接口

独立包位于 `ros2_ws/src/xuk_d435i_bridge` 与 `ros2_ws/src/xuk_d435i_interfaces`，**只发布相机能力，不接入也不修改任何现有机械臂或抓取工作空间**。

消息 `Object6D` / `Object6DArray`，服务 `CalibrationCommand` / `GetJson` / `SetControl` / `SetStream`。话题、字段、构建与调用示例见 [`docs/ROS2_FOXY_API.md`](docs/ROS2_FOXY_API.md)。

## 机械臂适配

相机模块只读末端位姿。机械臂侧实现 `RobotPoseProvider.latest()` 并返回米制 `base_T_tool` 即可，约定见 [`docs/ROBOT_POSE_ADAPTER.md`](docs/ROBOT_POSE_ADAPTER.md)。

在 ROS topic、消息类型和字段确定之前，GUI 保留手动位姿入口，**不会猜测消息定义**。

## 自定义物品训练

提供鼠标画框的可视化标注工具、`YoloAnnotationProject` 数据集 API、`YoloTrainingService` 训练 API、`d435i-yolo-train` 命令行和 GUI 后台训练入口。框住物体后自动计算并保存 YOLO 中心点与宽高，无需人工填坐标；训练前会校验图像、标签、类别编号与归一化坐标。详见 [`docs/MODEL_TRAINING.md`](docs/MODEL_TRAINING.md)。

## 已知限制

完整报告见 [`docs/TEST_REPORT_UBUNTU.md`](docs/TEST_REPORT_UBUNTU.md)，测试环境为 Intel Core i3-8100 / 无 CUDA，全部推理在 CPU 上完成。

- 6D 位姿对球体、圆柱绕轴、对称件、遮挡堆叠存在不可观测或 180° 歧义（见上文第 2 节）。
- 通用检测/分割模型不直接预测类别级 6D，`depth_pca` 是几何近似。
- 当前默认 CPU 推理，帧率受 `frame_interval` 控制。

## Roadmap

以下为**规划中、尚未实现**的方向：

- [x] 眼在手上 5 种算法交叉验证
- [x] 眼在手外 SHAH/LI 联合求解
- [x] 无相机 mock 后端与完整自动化测试
- [x] 44 项实机参数命名空间化
- [ ] 类别级 6D Pose 插件（CAD / 关键点 / 专用网络）
- [ ] GPU / TensorRT `.engine` 推理加速的实测基准
- [ ] 头部为运动关节时，由机械臂侧提供 `base_T_head` 扩展固定外参
- [ ] 多相机同时在线与统一坐标系管理

## 文档

| 文档 | 内容 |
| --- | --- |
| [`README_CN.md`](README_CN.md) | 原始完整中文说明（含 6D 定义、坐标系链、逐项功能） |
| [`docs/API_REFERENCE_CN.md`](docs/API_REFERENCE_CN.md) | 全部 Python API |
| [`docs/ROS2_FOXY_API.md`](docs/ROS2_FOXY_API.md) | ROS 2 话题、服务、消息 |
| [`docs/MODEL_TRAINING.md`](docs/MODEL_TRAINING.md) | 标注与训练完整步骤 |
| [`docs/ROBOT_POSE_ADAPTER.md`](docs/ROBOT_POSE_ADAPTER.md) | 机械臂位姿接入约定 |
| [`docs/TEST_REPORT_UBUNTU.md`](docs/TEST_REPORT_UBUNTU.md) | 实机测试报告 |

## 许可

本项目基于 [MIT License](LICENSE) 开源。
Ultralytics YOLOv8 及其官方权重遵循 AGPL-3.0，不在本仓库内分发。
