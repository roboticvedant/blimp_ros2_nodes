import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import csv
import math
from datetime import datetime

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        # Publisher and subscriber
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)

        # Timer for publishing commands
        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.publish_command)

        # Data logging setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f'data_{timestamp}.csv'
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Angular_Z_Command', 'Yaw_Angle', 'Yaw_Rate'])

        # Variables
        self.angular_z_command = 0.0
        self.last_yaw = 0.0
        self.last_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.init_time = self.last_time
        self.get_logger().info(f"Logging data to {self.log_file}")



    def publish_command(self):
        # Define the current time
        current_time = self.get_clock().now().seconds_nanoseconds()[0] - self.init_time
        print(current_time)
        # Choose input signal based on elapsed time
        if current_time < 10:
            # Step Input (constant command)
            self.angular_z_command = 0.5
        elif 10 <= current_time < 20:
            # Ramp Input (linearly increasing command)
            self.angular_z_command = 0.05 * (current_time - 10)  # Adjust slope as needed
        elif 20 <= current_time < 30:
            # Sinusoidal Input (oscillatory command)
            frequency = 0.5  # Frequency in Hz
            self.angular_z_command = 0.5 * math.sin(2 * math.pi * frequency * (current_time - 20))
        elif 30 <= current_time < 40:
            # PRBS (Pseudo-Random Binary Sequence)
            # Switch between -0.5 and 0.5 every 2 seconds
            self.angular_z_command = 0.5 if int((current_time - 30) / 2) % 2 == 0 else -0.5
        else:
            # Reset or hold the last command for additional testing
            self.angular_z_command = 0.0

        # Publish the command
        twist_msg = Twist()
        twist_msg.angular.z = self.angular_z_command
        self.cmd_vel_pub.publish(twist_msg)

        # Log the command
        self.get_logger().info(f"Published angular.z: {self.angular_z_command}")


    def imu_callback(self, msg):
        # Extract timestamp from the IMU message header
        timestamp = msg.header.stamp
        current_time_in_sec = timestamp.sec + timestamp.nanosec * 1e-9  # Convert to seconds

        # Convert quaternion to yaw
        yaw = self.quaternion_to_yaw(msg.orientation)

        yaw_rate = msg.angular_velocity.z

        # Log data
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time_in_sec,
                            self.angular_z_command,
                            yaw,
                            yaw_rate])

        # Update for next callback
        self.last_yaw = yaw
        self.last_time = current_time_in_sec


    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw (in radians)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    collector = DataCollector()

    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Data collection stopped.")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
