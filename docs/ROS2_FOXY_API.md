# ROS2 Foxy 相机接口

本目录提供的是**独立相机接口**，不修改、不依赖现有机械臂或抓取包。集成人员可选择只订阅
标准图像话题，也可直接订阅完整六维结果。运行环境为 Ubuntu 20.04 / ROS2 Foxy。

## 为什么有两个进程

相机、RealSense 与 Ultralytics 已安装在项目 Python 3.10 虚拟环境；主机上的 ROS2 Foxy
二进制扩展使用其自身 Python ABI，不能直接导入这个虚拟环境。因此采用本机双进程：

1. `d435i-camera-gateway`：Python 3.10，独占 D435i，完成 RGB-D、YOLO、6D 与标定。
2. `xuk_d435i_bridge`：ROS2 Foxy 环境，发布 ROS 消息和服务。

两者只通过 `127.0.0.1:17630/17631` 通信，不向局域网开放。GUI 与 gateway 不能同时占用
同一台 D435i；运行 ROS 接口时应先关闭 GUI。

## 构建

```bash
cd /home/user/d435i_camera_module
source /opt/ros/foxy/setup.bash
mkdir -p ros2_build_ws/src
ln -sfn "$PWD/ros2_ws/src/xuk_d435i_interfaces" ros2_build_ws/src/xuk_d435i_interfaces
ln -sfn "$PWD/ros2_ws/src/xuk_d435i_bridge" ros2_build_ws/src/xuk_d435i_bridge
cd ros2_build_ws
colcon build --symlink-install --packages-select xuk_d435i_interfaces xuk_d435i_bridge
```

## 启动

终端 1（相机/YOLO）：

```bash
cd /home/user/d435i_camera_module
source .venv/bin/activate
d435i-camera-gateway --config config/default.yaml
```

无相机联调时可在上述命令末尾加 `--mock`，话题中会持续输出模拟的水瓶和杯子。

终端 2（ROS2）：

```bash
source /opt/ros/foxy/setup.bash
source /home/user/d435i_camera_module/ros2_build_ws/install/setup.bash
ros2 launch xuk_d435i_bridge d435i_bridge.launch.py
```

## 发布话题

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/d435i/color/image_raw` | `sensor_msgs/msg/Image` | BGR8 彩色图 |
| `/d435i/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | `32FC1` 米制深度，已对齐彩色图 |
| `/d435i/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 当前彩色流内参 |
| `/d435i/annotated/image_raw` | `sensor_msgs/msg/Image` | 框、坐标轴、中文名和六维数据叠加图 |
| `/d435i/objects_6d` | `xuk_d435i_interfaces/msg/Object6DArray` | 完整物体数组 |
| `/yolov8/detections` | `vision_msgs/msg/Detection2DArray` | 可选兼容输出；bbox 为 0–1 归一化坐标 |

`Object6D` 的字段：英文类别、中文类别、类别 ID、置信度、像素框 `xyxy`、ROS Pose、
三维框尺寸 `dimensions_m`、`rx_deg/ry_deg/rz_deg`、坐标系、有效标志、说明和深度。
位置和尺寸单位为米，角度单位为度；
未完成手眼变换时坐标系为 `camera_color_optical_frame`。

## 服务接口

| 服务 | 类型 | 用途 |
|---|---|---|
| `/d435i/start` | `std_srvs/srv/Trigger` | 启动相机处理 |
| `/d435i/stop` | `std_srvs/srv/Trigger` | 停止相机处理 |
| `/d435i/snapshot` | `std_srvs/srv/Trigger` | 保存彩色、深度和元数据 |
| `/d435i/get_json` | `xuk_d435i_interfaces/srv/GetJson` | 查询 `status/profiles/controls` |
| `/d435i/set_stream` | `xuk_d435i_interfaces/srv/SetStream` | 分别设置 RGB/Depth 分辨率和帧率、IMU |
| `/d435i/set_control` | `xuk_d435i_interfaces/srv/SetControl` | 设置 `color.* / depth.* / motion.*` 硬件参数 |
| `/d435i/calibration` | `xuk_d435i_interfaces/srv/CalibrationCommand` | 内参、眼在手上、眼在手外标定 |

查询接口示例：

```bash
ros2 service call /d435i/get_json xuk_d435i_interfaces/srv/GetJson \
  "{operation: profiles}"
ros2 service call /d435i/get_json xuk_d435i_interfaces/srv/GetJson \
  "{operation: controls}"
```

分别修改彩色和深度 profile：

```bash
ros2 service call /d435i/set_stream xuk_d435i_interfaces/srv/SetStream \
  "{color_width: 1280, color_height: 720, color_fps: 30, depth_width: 848, depth_height: 480, depth_fps: 30, enable_imu: false, persist: true}"
```

修改硬件参数：

```bash
ros2 service call /d435i/set_control xuk_d435i_interfaces/srv/SetControl \
  "{name: color.brightness, value: 5.0}"
ros2 service call /d435i/set_control xuk_d435i_interfaces/srv/SetControl \
  "{name: depth.laser_power, value: 150.0}"
```

标定统一使用 `reset -> capture（多次） -> solve`。眼在手上/眼在手外的每次 `capture`
必须由机械臂侧填写同步的 `base_to_tool`；相机模块不会控制机械臂运动。

## 给接入人员的约定

- 首选订阅 `/d435i/objects_6d`，不要从叠加图 OCR 六维数据。
- 抓取前必须检查 `Object6D.valid == true` 和 `frame_id`。
- 对称瓶体绕自身轴的角度可能存在不可观测/180° 歧义，抓取策略应允许该自由度。
- 标准深度为米制 `32FC1`；无效像素可能是 `0` 或非有限值。
- 兼容 `/yolov8/detections` 仅承担 2D 接口，完整中文名与 6D 以自定义消息为准。
