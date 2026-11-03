# Ubuntu 20.04 验证报告

验证日期：2026-08-14  
运行主机：Ubuntu 20.04 桌面（测试机）  
运行目录：`/home/user/d435i_camera_module`

## 环境

- Ubuntu 20.04.6 LTS，Python 3.10.11。
- CPU：Intel Core i3-8100，4 核，无可用 CUDA GPU。
- Intel RealSense D435i，固件 `5.15.1.55`，USB 3.2。
- librealsense 系统包 2.56.5；项目虚拟环境 `pyrealsense2 2.58.3.10794`。
- OpenCV 4.11.0，NumPy 1.26.4，Ultralytics 8.3.168，PyQt5，Pillow 12.0.0。
- `/usr/local` Python 缺少 `_lzma`，项目通过 `backports.lzma 0.0.14` 做局部兼容，
  不修改系统 Python。

## 已通过

- `python -m compileall -q src`：通过。
- 三个 Bash 运行/安装脚本 `bash -n`：通过。
- `pytest`：10 个测试全部通过，包括坐标几何、PCA 姿态、眼在手上与眼在手外合成
  真值标定、中文 6D 叠加、配置、RGB-D 模拟全管线、快照和 PLY 点云。
- D435i 枚举、启停、30 帧连续采集、彩色/深度对齐：通过。
- 分辨率/帧率共同 profile 查询：真实 D435i 返回 19 组支持配置。
- 独立 profile 查询：RGB BGR8 返回 32 组，Depth Z16 返回 32 组；已实机同时启动
  RGB 1280×720@30 与 Depth 848×480@30。深度对齐到彩色后两路输出均为 1280×720。
- 运行中安全重配置：已从 848×480@30 切换到 640×480@15 + IMU，随后恢复默认；通过。
- GUI 分辨率/帧率/IMU 控件：RGB 32 组与 Depth 32 组分别加载，当前默认项正确；通过。
- 真实流分辨率 848×480，实测 29.53 FPS（配置 30 FPS）。
- 当前彩色内参：`fx=603.4271, fy=602.3505, cx=435.4451, cy=243.7277`。
- 曝光、增益、自动曝光、发射器、激光功率、视觉预设枚举：通过。
- 完整参数动态枚举：44 项；`color.exposure` 与 `depth.exposure` 已分离。RGB 16 项、
  Depth 25 项、Motion 3 项；安全项同值写回通过，`depth.depth_units` 保护拒写通过。
- 快照（PNG + 米制 NPY + JSON）、出厂内参 JSON、PLY 点云导出：通过。
- D435i IMU 自动 profile 选择：通过；30/30 帧取得加速度和陀螺仪数据。
- PyQt5 无头完整启动：通过；5 个功能页、模拟 bottle/cup、实时 6D 表格和叠加均正常。
- GUI 预览保存于远端 `data/gui_mock_preview.png`。
- 官方 `models/yolov8n.pt` 已下载并通过模块 `YoloDetector` 在旧项目水瓶图上验证：
  阈值 0.30 时返回 COCO class 39 `bottle`，置信度 0.408；首次 CPU 推理约 999 ms，
  预热后连续两帧分别约 48.9/49.5 ms。
- 中文叠加图已在 Ubuntu 的 Noto Sans CJK 字体下实测，左上角和框旁均显示
  “物品中文名称 + X/Y/Z/Rx/Ry/Rz”。

## 现场待处理项

真实彩色测试图几乎是整幅失焦蓝色；深度有效像素约 5.43%，中值约 0.173 m，接近
D435i 近距离下限。推断镜头前存在蓝色保护膜/遮挡物，或镜头正贴近蓝色表面。

因此以下项目不能在当前现场画面下声称已通过：

1. 当前 D435i 现场画面中的真实水瓶检测效果（离线水瓶图已通过）。
2. 普通框/分割 mask 下的真实物品 6D 稳定性。
3. 8×11、15 mm 棋盘的真实内参、眼在手上及头部眼在手外标定误差。
4. 机械臂 base 坐标输出（ROS topic/消息由机械臂开发人员适配）。

继续验收前应：移除镜头保护膜或遮挡；把棋盘/物体放在约 0.25–1.5 m；改善纹理与光照；
确认 Ultralytics 许可证适合项目交付方式；如需更稳的轮廓点云，可下载
`yolov8n-seg.pt`，或提供最终自定义模型。
