import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import MagneticField
from KinematicEstimation import SensorDataFilter


class MagnetometerController(Node):
    def __init__(self):
        super().__init__('magnetometer_controller')

        # Parameters for the target magnetometer range
        self.target_z = -25.0  # Desired magnetometer z value
        self.tolerance = 5.0  # Tolerance range (±)

        # Control parameters
        self.max_cmd_vel = 1.0  # Maximum linear.z velocity
        self.kp = 0.1  # Proportional gain for controller

        # Filter for the magnetometer data
        self.sensor_filter = SensorDataFilter()

        # Subscriber to the magnetometer topic
        self.mag_subscriber = self.create_subscription(
            MagneticField,
            '/magnetometer/data',  # Replace with your magnetometer topic
            self.magnetometer_callback,
            10
        )

        # Publisher for cmd_vel
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Last published Twist message
        self.last_twist = Twist()

    def magnetometer_callback(self, msg: MagneticField):
        # Extract and filter the magnetometer z-axis data
        raw_magnetometer = {'x': msg.magnetic_field.x, 'y': msg.magnetic_field.y, 'z': msg.magnetic_field.z}
        filtered_magnetometer = self.sensor_filter.filter_magnetometer(raw_magnetometer)

        # Calculate the error for z-axis
        error = self.target_z - filtered_magnetometer['z']

        # If the magnetometer z-value is within the desired range, stop adjustments
        if abs(error) <= self.tolerance:
            self.last_twist.linear.z = 0.0
        else:
            # Proportional control (inverted logic to account for inverse relationship)
            self.last_twist.linear.z = -self.kp * error  # Negate the error to correct direction
            # Limit the cmd_vel to max_cmd_vel
            self.last_twist.linear.z = max(-self.max_cmd_vel, min(self.last_twist.linear.z, self.max_cmd_vel))

        # Publish the cmd_vel message
        self.cmd_vel_publisher.publish(self.last_twist)

        # Log the filtered value and control action
        self.get_logger().info(
            f"Magnetometer Z: {filtered_magnetometer['z']:.2f}, Error: {error:.2f}, Commanded linear.z: {self.last_twist.linear.z:.2f}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = MagnetometerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
