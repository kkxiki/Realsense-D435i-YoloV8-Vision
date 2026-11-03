from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("xuk_d435i_bridge"), "config", "bridge.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="xuk_d435i_bridge",
                executable="bridge_node",
                name="xuk_d435i_bridge",
                output="screen",
                parameters=[config],
            )
        ]
    )
