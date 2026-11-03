# xuk_d435i_bridge (ROS2 Foxy)

This package publishes the independent `d435i_camera_module` through ROS2 Foxy.
It does not depend on, modify, or launch any robot-arm/grasp package.

The camera/YOLO process uses Python 3.10. The Foxy bridge uses the Python ABI supplied
with Foxy. They communicate only through localhost TCP (`17630` control, `17631` stream).

See `d435i_camera_module/docs/ROS2_FOXY_API.md` for build, launch, topics and services.
