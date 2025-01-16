from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    return LaunchDescription([
        # USB Camera 1
        ExecuteProcess(
            cmd=['ros2', 'run', 'usb_cam', 'usb_cam_node_exe', 
                 '--ros-args', '-p', 'video_device:=/dev/video0', 
                 '-r', 'image_raw:=/camera1/image_raw'],
            output='screen'
        ),
        
        # # USB Camera 2
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'usb_cam', 'usb_cam_node_exe', 
        #          '--ros-args', '-p', 'video_device:=/dev/video1', 
        #          '-r', 'image_raw:=/camera2/image_raw'],
        #     output='screen'
        # ),

        # AprilTag detector node for camera 1
        Node(
            package='aprilTag_test',
            executable='tag_detect',
            name='tag_detect_camera1',
            parameters=[{
                'camera_name': 'camera1',
                'camera_topic': '/camera1/image_raw',
                'calib_file': os.path.expanduser('~/Desktop/ost.yaml')
            }]
        ),

        # # AprilTag detector node for camera 2
        # Node(
        #     package='aprilTag_test',
        #     executable='tag_detect',
        #     name='tag_detect_camera2',
        #     parameters=[{
        #         'camera_name': 'camera2',
        #         'camera_topic': '/camera2/image_raw'
        #     }]
        # ),

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