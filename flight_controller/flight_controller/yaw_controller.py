import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Imu
from geometry_msgs.msg import Twist
from math import atan2, pi, cos, sin
import matplotlib.pyplot as plt
import time


class YawControllerWithPID(Node):
    def __init__(self):
        super().__init__('yaw_controller_with_pid')

        # Parameters for yaw control
        self.current_yaw = 0.0  # Current yaw in radians
        self.target_yaw = 0.0  # Target yaw in radians
        self.kp = 0.6  # Proportional gain
        self.ki = 0.0  # Integral gain
        self.kd = 0.05  # Derivative gain
        self.max_angular_z = 1.0  # Max angular velocity
        self.joystick_scale = 0.1  # Joystick input scale for target yaw adjustment
        self.error_tolerance = 0.25  # Allowable error range in radians (dead zone)

        # PID terms
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()
        self.start_time = time.time()  # Start time for tracking elapsed time

        # Joystick axes mapping
        self.axis_angular_z = 0  # Default axis for angular.z (Left stick horizontal axis)

        # X and Y linear axes for future use
        self.axis_linear_x = 1
        self.axis_linear_z = 3
        self.scale_linear_x = 1.0
        self.scale_linear_z = 1.0

        # Flag to track joystick activity
        self.joystick_active = False

        # Subscribers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)

        # Publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer for regular control updates
        self.control_timer = self.create_timer(0.1, self.control_yaw)

        # Plotting setup
        self.fig, self.ax_unit_circle = plt.subplots(figsize=(6, 6))
        self.ax_unit_circle.set_aspect('equal')
        self.ax_unit_circle.set_xlim(-1.5, 1.5)
        self.ax_unit_circle.set_ylim(-1.5, 1.5)
        self.ax_unit_circle.grid(True)

        # Plot for system response
        self.fig_response, self.ax_response = plt.subplots(figsize=(10, 6))
        self.time_data = []
        self.yaw_data = []
        self.target_data = []

        self.get_logger().info("YawControllerWithPID initialized.")

    def joy_callback(self, msg: Joy):
        """
        Adjust target yaw incrementally based on joystick input.
        """
        joystick_input = msg.axes[self.axis_angular_z]  # Get the angular.z axis input

        self.target_yaw += joystick_input * self.joystick_scale

        #X and Z linear axes for future use
        self.linear_x_vel = msg.axes[self.axis_linear_x] * self.scale_linear_x
        self.linear_z_vel = msg.axes[self.axis_linear_z] * self.scale_linear_z

        # Normalize the target yaw to stay within [-π, π]
        self.target_yaw = self.normalize_angle(self.target_yaw)

        # Set joystick activity flag to True
        self.joystick_active = True

        self.get_logger().info(f"Joystick input: {joystick_input}, Updated target yaw: {self.target_yaw:.2f} rad")

    def imu_callback(self, msg: Imu):
        """
        Update current yaw from IMU quaternion data.
        """
        q = msg.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.get_logger().info(f"Current yaw: {self.current_yaw:.2f} rad")
        self.update_plot()
        self.update_response_plot()

    def control_yaw(self):
        """
        Calculate and publish cmd_vel to adjust yaw towards the target, only if joystick is active.
        """
        if not self.joystick_active:
            # Do not publish anything if the joystick is inactive
            self.get_logger().info("Joystick inactive, no cmd_vel published.")
            return

        # Calculate yaw error
        yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)

        # Check if the error is within the tolerance range (dead zone)
        if abs(yaw_error) <= self.error_tolerance:
            self.get_logger().info(f"Yaw error {yaw_error:.2f} within tolerance, no adjustment needed.")
            self.reset_pid()
            # Publish cmd_vel message for linear velocities
            twist_msg = Twist()

            twist_msg.linear.x = self.linear_x_vel
            twist_msg.linear.z = self.linear_z_vel
            
            self.cmd_vel_pub.publish(twist_msg)
            return

        # Calculate time delta
        current_time = time.time()
        time_delta = current_time - self.previous_time
        self.previous_time = current_time

        # PID calculations
        # Proportional term
        proportional = self.kp * yaw_error

        # Integral term
        self.integral += yaw_error * time_delta
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (yaw_error - self.previous_error) / time_delta
        self.previous_error = yaw_error

        # PID output
        angular_z = proportional + integral + derivative

        # Clamp angular velocity to max_angular_z
        angular_z = max(-self.max_angular_z, min(angular_z, self.max_angular_z))

        # Publish cmd_vel message
        twist_msg = Twist()
        twist_msg.angular.z = angular_z

        twist_msg.linear.x = self.linear_x_vel
        twist_msg.linear.z = self.linear_z_vel
        
        self.cmd_vel_pub.publish(twist_msg)

        self.get_logger().info(
            f"Yaw error: {yaw_error:.2f}, P: {proportional:.2f}, I: {integral:.2f}, D: {derivative:.2f}, Commanded angular.z: {angular_z:.2f}"
        )

    def update_plot(self):
        """
        Update the unit circle plot with current yaw, target yaw, and tolerance zone.
        """
        self.ax_unit_circle.clear()
        self.ax_unit_circle.set_aspect('equal')
        self.ax_unit_circle.set_xlim(-1.5, 1.5)
        self.ax_unit_circle.set_ylim(-1.5, 1.5)
        self.ax_unit_circle.grid(True)

        # Draw unit circle
        circle = plt.Circle((0, 0), 1, color='blue', fill=False)
        self.ax_unit_circle.add_artist(circle)

        # Draw tolerance zone
        tolerance_arc_start = self.target_yaw - self.error_tolerance
        tolerance_arc_end = self.target_yaw + self.error_tolerance
        angles = [tolerance_arc_start + i * (tolerance_arc_end - tolerance_arc_start) / 50 for i in range(51)]
        tolerance_x = [cos(angle) for angle in angles]
        tolerance_y = [sin(angle) for angle in angles]
        self.ax_unit_circle.fill(tolerance_x, tolerance_y, color='green', alpha=0.3, label="Tolerance Zone")

        # Draw target yaw
        target_x = cos(self.target_yaw)
        target_y = sin(self.target_yaw)
        self.ax_unit_circle.arrow(0, 0, target_x * 1.2, target_y * 1.2, head_width=0.1, color='red', label="Target Yaw")

        # Draw current yaw
        current_x = cos(self.current_yaw)
        current_y = sin(self.current_yaw)
        self.ax_unit_circle.arrow(0, 0, current_x * 1.1, current_y * 1.1, head_width=0.1, color='orange', label="Current Yaw")

        # Add legend
        self.ax_unit_circle.legend()

        # Update plot
        plt.pause(0.01)

    def update_response_plot(self):
        """
        Update the system response plot over time.
        """
        # Track elapsed time
        elapsed_time = time.time() - self.start_time
        self.time_data.append(elapsed_time)
        self.yaw_data.append(self.current_yaw)
        self.target_data.append(self.target_yaw)

        # Plot the yaw response over time
        self.ax_response.clear()
        self.ax_response.plot(self.time_data, self.yaw_data, label="Current Yaw", color='orange')
        self.ax_response.plot(self.time_data, self.target_data, label="Target Yaw", color='red', linestyle='--')
        self.ax_response.set_xlabel("Time (s)")
        self.ax_response.set_ylabel("Yaw (rad)")
        self.ax_response.set_title(f"PID Response (Kp={self.kp}, Ki={self.ki}, Kd={self.kd})")
        self.ax_response.legend()
        self.ax_response.grid()
        plt.pause(0.01)

    def reset_pid(self):
        """
        Reset PID terms.
        """
        self.integral = 0.0
        self.previous_error = 0.0

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        """
        Convert quaternion to yaw angle in radians.
        """
        # Yaw calculation from quaternion
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return atan2(t3, t4)

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to the range [-π, π].
        """
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = YawControllerWithPID()
    plt.ion()  # Turn on interactive mode for Matplotlib
    plt.show()  # Show the plot
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
