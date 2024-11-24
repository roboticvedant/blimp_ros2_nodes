# FILE: launch/joystick_control.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joystick_control',
            executable='joystick_to_cmd_vel',
            name='joystick_to_cmd_vel',
            output='screen',
        ),
    ])