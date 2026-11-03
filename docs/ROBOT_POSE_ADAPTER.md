# 机械臂末端位姿适配约定

相机模块不依赖具体机械臂消息。机械臂开发人员只需实现
`d435i_camera.robot_pose.RobotPoseProvider`，返回带时间戳的 `base_T_tool`。

## 坐标约定

- `base_T_tool`：把 tool 坐标转换到 base 坐标的 4×4 齐次矩阵。
- 平移进入相机模块前必须换算为米。
- 旋转矩阵必须为右手系、正交且行列式为 +1。
- 时间戳必须能用于判断新鲜度，默认超过 0.5 秒拒绝用于实时变换。
- 标定和运行时的 `tool` 必须是同一个法兰/末端坐标系。

## 两种安装方式

- 腕部相机（eye-in-hand）：机器人侧持续提供 `base_T_tool`，模块使用
  `base_T_tool × tool_T_camera` 得到实时 `base_T_camera`。
- 头部固定相机（eye-to-hand）：标定得到并直接加载固定的 `base_T_camera`，识别运行时
  不需要末端实时位姿。
- 如果头部本身会转动，应由机器人侧额外提供 `base_T_head`，并使用
  `base_T_head × head_T_camera`；不能把会运动的头部当作固定基座相机。

最小适配示例：

```python
import time
from d435i_camera.robot_pose import CallbackRobotPoseProvider, TimedRobotPose

def read_latest_pose():
    # TODO: 从 ROS 订阅缓存中取值并构造 base_T_tool（平移单位 m）
    return TimedRobotPose(base_T_tool=matrix_4x4, timestamp_s=time.time(), source="ros")

module.set_robot_pose_provider(CallbackRobotPoseProvider(read_latest_pose))
```

远端 ROS Foxy 使用 Python 3.8，而 GUI 使用 Python 3.10。若 `rclpy` 不能在 GUI
虚拟环境导入，可在 ROS Python 3.8 进程里订阅并通过本机 UDP 发送 JSON，GUI 侧使用
`UdpRobotPoseProvider`。JSON 协议：

```json
{
  "timestamp_s": 1700000000.0,
  "translation": [100.0, 200.0, 300.0],
  "translation_unit": "mm",
  "rotation": [0.0, 0.0, 1.57],
  "rotation_type": "rpy",
  "angle_unit": "rad"
}
```

拿到确切 ROS topic、消息类型和字段后，再实现对应的订阅桥；相机模块内部无需修改。
