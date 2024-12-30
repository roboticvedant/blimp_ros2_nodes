# BLIMP ROS2 Nodes

A ROS2 package for blimp flight control and joystick integration, providing nodes for flight control, yaw management, and kinematic state estimation. Still under development and testing. Developed for the blimp v1 @ SML-MSU.

## Features

- Joystick/controller integration with ROS2
- Flight controller implementation
- Yaw control system
- Dead reckoning capabilities
- Kinematic state estimation
- Low-pass filtering

## Prerequisites

- ROS2 (tested with ROS2 Humble)
- Python 3.x
- PS4 Controller (for joystick control)
- [Bash Helpers](https://github.com/roboticvedant/bash_helpers.git) (recommended)

## Installation

1. Clone the repository into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone [repository-url]
```

2. Install dependencies:
```bash
cd ~/ros2_ws
colcon build
```

## Usage

To connect a PS4 controller and launch the teleop node:
```bash
ros2 launch teleop_twist_joy teleop-launch.py
```

## Package Structure

- `flight_controller/`: Core flight control implementation ros pkg
- `joystick_control/`: Joystick input processing and command velocity mapping ros pkg
- `KinematicEstimation/`: State estimation and dead reckoning dependency used by different packages


## Contributing

[Your contribution guidelines will go here]

## License
MIT License

## Contact

[Vedant K. Naik](mailto:vnaik792014@gmail.com)