import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from .KinematicEstimation import DeadReckoning
from .KinematicEstimation import SensorDataFilter

class IMUTfPublisher(Node):
    def __init__(self):
        super().__init__('imu_tf_publisher')

        # Subscribers
        self.subscription_imu = self.create_subscription(
            Imu,
            '/imu/data',  # Replace with your IMU topic
            self.imu_callback,
            1
        )
        self.subscription_mag = self.create_subscription(
            MagneticField,
            '/magnetometer/data',  # Replace with your magnetometer topic
            self.mag_callback,
            1
        )

        # TF2 broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Frames
        self.world_frame = "world"
        self.imu_frame = "imu_frame"

        # Dead Reckoning instance
        self.dead_reckoning = DeadReckoning()
        self.sensor_filter = SensorDataFilter()  # Initialize with a smoothing factor

        # Real-time plotting setup
        self.plot_length = 100  # Number of data points to retain
        self.linear_acc_x = deque(maxlen=self.plot_length)
        self.linear_acc_y = deque(maxlen=self.plot_length)
        self.linear_acc_z = deque(maxlen=self.plot_length)
        self.angular_vel_x = deque(maxlen=self.plot_length)
        self.angular_vel_y = deque(maxlen=self.plot_length)
        self.angular_vel_z = deque(maxlen=self.plot_length)
        self.magnetometer_x = deque(maxlen=self.plot_length)
        self.magnetometer_y = deque(maxlen=self.plot_length)
        self.magnetometer_z = deque(maxlen=self.plot_length)
        self.time_stamps = deque(maxlen=self.plot_length)

        # Initialize plotting
        plt.ion()
        self.fig_acc = plt.figure(figsize=(8, 6))
        self.fig_vel = plt.figure(figsize=(8, 6))
        self.fig_mag = plt.figure(figsize=(8, 6))
        self.ax_acc = self.fig_acc.add_subplot(1, 1, 1)
        self.ax_vel = self.fig_vel.add_subplot(1, 1, 1)
        self.ax_mag = self.fig_mag.add_subplot(1, 1, 1)

    def imu_callback(self, msg: Imu):
        # Extract linear acceleration and angular velocity from IMU message
        linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        filtered_linear_acceleration = self.sensor_filter.filter_linear_acceleration(linear_acceleration)
        filtered_angular_velocity = self.sensor_filter.filter_angular_velocity(angular_velocity)

        # Extract timestamp from IMU message header
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update position using Dead Reckoning
        position = self.dead_reckoning.update(
            np.array([filtered_linear_acceleration['x'], filtered_linear_acceleration['y'], filtered_linear_acceleration['z']]),
            timestamp
        )

        # Update data for plotting
        self.time_stamps.append(timestamp)
        self.linear_acc_x.append(filtered_linear_acceleration['x'])
        self.linear_acc_y.append(filtered_linear_acceleration['y'])
        self.linear_acc_z.append(filtered_linear_acceleration['z'])
        self.angular_vel_x.append(filtered_angular_velocity['x'])
        self.angular_vel_y.append(filtered_angular_velocity['y'])
        self.angular_vel_z.append(filtered_angular_velocity['z'])

        # Update plots
        self.update_plot_acc()
        self.update_plot_vel()

        # Check if the position has changed significantly
        movement_threshold = 0.01  # Adjust for your use case
        if np.linalg.norm(position) > movement_threshold:
            self.broadcast_tf(position, msg.orientation)  # Pass orientation to broadcast_tf

    def mag_callback(self, msg: MagneticField):
        # Extract magnetometer data
        magnetometer = np.array([
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        ])

        filtered_magnetometer = self.sensor_filter.filter_magnetometer(magnetometer)

        # Update data for plotting
        self.magnetometer_x.append(filtered_magnetometer['x'])
        self.magnetometer_y.append(filtered_magnetometer['y'])
        self.magnetometer_z.append(filtered_magnetometer['z'])

        # Update plot
        self.update_plot_mag()

    def broadcast_tf(self, position, orientation):
        """Broadcast the current transform from world to IMU frame."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()  # Use current ROS time
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.imu_frame

        # Set translation (current position)
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        # Set rotation (from IMU orientation)
        transform.transform.rotation.x = orientation.x
        transform.transform.rotation.y = orientation.y
        transform.transform.rotation.z = orientation.z
        transform.transform.rotation.w = orientation.w

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    def update_plot_acc(self):
        """Update real-time plot for linear acceleration."""
        self.ax_acc.clear()
        self.ax_acc.plot(self.time_stamps, self.linear_acc_x, label="X")
        self.ax_acc.plot(self.time_stamps, self.linear_acc_y, label="Y")
        self.ax_acc.plot(self.time_stamps, self.linear_acc_z, label="Z")
        self.ax_acc.legend()
        self.ax_acc.set_title("Linear Acceleration")
        self.ax_acc.set_xlabel("Time (s)")
        self.ax_acc.set_ylabel("Acceleration (m/s^2)")
        self.ax_acc.grid()
        self.fig_acc.canvas.draw()
        self.fig_acc.canvas.flush_events()

    def update_plot_vel(self):
        """Update real-time plot for angular velocity."""
        self.ax_vel.clear()
        self.ax_vel.plot(self.time_stamps, self.angular_vel_x, label="X")
        self.ax_vel.plot(self.time_stamps, self.angular_vel_y, label="Y")
        self.ax_vel.plot(self.time_stamps, self.angular_vel_z, label="Z")
        self.ax_vel.legend()
        self.ax_vel.set_title("Angular Velocity")
        self.ax_vel.set_xlabel("Time (s)")
        self.ax_vel.set_ylabel("Velocity (rad/s)")
        self.ax_vel.grid()
        self.fig_vel.canvas.draw()
        self.fig_vel.canvas.flush_events()

    def update_plot_mag(self):
        """Update real-time plot for magnetometer data."""
        self.ax_mag.clear()
        self.ax_mag.plot(range(len(self.magnetometer_x)), self.magnetometer_x, label="X")
        self.ax_mag.plot(range(len(self.magnetometer_y)), self.magnetometer_y, label="Y")
        self.ax_mag.plot(range(len(self.magnetometer_z)), self.magnetometer_z, label="Z")
        self.ax_mag.legend()
        self.ax_mag.set_title("Magnetometer Data")
        self.ax_mag.set_xlabel("Time (samples)")
        self.ax_mag.set_ylabel("Magnetic Field (uT)")
        self.ax_mag.grid()
        self.fig_mag.canvas.draw()
        self.fig_mag.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = IMUTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
