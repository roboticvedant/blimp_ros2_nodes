from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare the debug flag argument
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',  # Default to no debug
        description='Enable debug mode'
    )

    # Get the value of the debug flag
    debug_flag = LaunchConfiguration('debug')

    return LaunchDescription([
        debug_arg,

        # USB Camera 1
        ExecuteProcess(
            cmd=['ros2', 'run', 'usb_cam', 'usb_cam_node_exe',
                 '--ros-args', '-p', 'video_device:=/dev/video0',
                 '-r', 'image_raw:=/camera1/image_raw'],
            output='screen'
        ),

        # AprilTag detector node for camera 1
        Node(
            package='aprilTag_test',
            executable='tag_detect',
            name='tag_detect_camera1',
            parameters=[{
                'camera_name': 'camera1',
                'camera_topic': '/camera1/image_raw',
                'calib_file': os.path.expanduser('~/Desktop/ost.yaml'),
                'tag_size': 0.100,
                'tag_family': 'tag25h9',
                'get_debug_image': debug_flag
            }]
        ),

        # Transform manager node
        Node(
            package='aprilTag_test',
            executable='transform_manager',
            name='transform_manager',
            parameters=[{
                'camera_names': ['camera1']
            }]
        )
    ])
