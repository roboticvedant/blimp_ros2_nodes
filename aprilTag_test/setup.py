from setuptools import find_packages, setup

package_name = 'aprilTag_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/april_tag_tool.launch.py']),
        ('share/' + package_name + '/launch', ['launch/camera_multi.launch.py']),
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
            'tag_detect = aprilTag_test.tag_detect:main',
            'transform_manager = aprilTag_test.transform_manager:main',
        ],
    },
)
