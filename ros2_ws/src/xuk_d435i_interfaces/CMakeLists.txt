cmake_minimum_required(VERSION 3.5)
project(xuk_d435i_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Object6D.msg"
  "msg/Object6DArray.msg"
  "srv/CalibrationCommand.srv"
  "srv/GetJson.srv"
  "srv/SetControl.srv"
  "srv/SetStream.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
