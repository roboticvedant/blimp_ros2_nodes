from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'flight_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name, 'KinematicEstimation'), glob('../KinematicEstimation/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vnaik792014',
    maintainer_email='vnaik792014@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'magnetometer_controller = flight_controller.controller:main',
            'yaw_controller_with_joystick = flight_controller.yaw_controller:main',
            'yaw_data_collector = flight_controller.collect_yaw_data:main',
        ],
    },
)
