import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from evdev import InputDevice, ecodes

class JoystickToCmdVelNode(Node):
    def __init__(self):
        super().__init__('joystick_to_cmd_vel_node')

        # Declare parameters for joystick axes and scaling
        self.declare_parameters(
            namespace='',
            parameters=[
                ('axis_linear_x', 1),   # Default: Left stick vertical axis
                ('axis_linear_z', 3),   # Default: Right trigger axis
                ('axis_angular_z', 0),  # Default: Left stick horizontal axis
                ('scale_linear_x', 1.0),  # Scale to match -1 to 1
                ('scale_linear_z', 1.0),
                ('scale_angular_z', 1.0)
            ]
        )

        # Retrieve parameter values
        self.axis_linear_x = self.get_parameter('axis_linear_x').value
        self.axis_linear_z = self.get_parameter('axis_linear_z').value
        self.axis_angular_z = self.get_parameter('axis_angular_z').value
        self.scale_linear_x = self.get_parameter('scale_linear_x').value
        self.scale_linear_z = self.get_parameter('scale_linear_z').value
        self.scale_angular_z = self.get_parameter('scale_angular_z').value

        # Initialize haptic feedback device
        self.haptic_device = self.get_haptic_device()

        # Subscribers and Publishers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("JoystickToCmdVelNode started")

    def joy_callback(self, msg: Joy):
        twist = Twist()
        # Map joystick axes to the required components
        twist.linear.x = self.clamp(msg.axes[self.axis_linear_x] * self.scale_linear_x, -1.0, 1.0)
        twist.linear.z = self.clamp(msg.axes[self.axis_linear_z] * self.scale_linear_z, -1.0, 1.0)
        twist.angular.z = self.clamp(msg.axes[self.axis_angular_z] * self.scale_angular_z, -1.0, 1.0)

        # Publish the Twist message
        self.cmd_vel_pub.publish(twist)

        # Trigger haptic feedback based on linear.x
        self.trigger_haptic_feedback(twist.linear.x)

        self.get_logger().debug(f"Published cmd_vel: {twist}")

    def trigger_haptic_feedback(self, intensity):
        """
        Trigger haptic feedback on the PS4 controller.
        Intensity should be between -1.0 and 1.0.
        """
        if not self.haptic_device:
            return

        # Normalize intensity to 0-255 for vibration
        vibration_intensity = int(abs(intensity) * 255)

        # Create force feedback event
        effect = ecodes.FF_RUMBLE
        event = ecodes.EV_FF
        strong_magnitude = vibration_intensity  # Strong vibration motor
        weak_magnitude = vibration_intensity  # Weak vibration motor

        try:
            self.haptic_device.write_event(ecodes.InputEvent(event, effect, [strong_magnitude, weak_magnitude]))
            self.haptic_device.syn()
        except Exception as e:
            self.get_logger().error(f"Failed to send haptic feedback: {e}")

    def get_haptic_device(self):
        """
        Find and return the input device for the PS4 controller.
        """
        try:
            devices = [InputDevice(path) for path in InputDevice.list_devices()]
            for device in devices:
                if "Sony Interactive Entertainment Wireless Controller" in device.name:
                    self.get_logger().info(f"Haptic feedback device found: {device.name}")
                    return device
        except Exception as e:
            self.get_logger().error(f"Failed to find haptic feedback device: {e}")
        return None

    @staticmethod
    def clamp(value, min_value, max_value):
        """Ensure the value is within the specified range."""
        return max(min(value, max_value), min_value)

def main(args=None):
    rclpy.init(args=args)
    node = JoystickToCmdVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
