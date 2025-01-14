import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_calibration_file = os.path.expanduser('~/Desktop/ost.yaml')

    return LaunchDescription([
        # AprilTag node
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_detector',
            output='screen',
            parameters=[
                {
                    'camera_info_url': f"file://{camera_calibration_file}",
                    'tag_family': 'tag25h9',
                    'tag_size': 0.1286,  # 128.6 mm in meters
                }
            ],
            remappings=[
                ('/image', '/image_raw'),  # Change '/image_raw' if your camera topic is different
                ('/camera_info', '/camera/camera_info'),
            ],
        ),
    ])
