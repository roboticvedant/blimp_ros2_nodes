#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
from pupil_apriltags import Detector
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('tag_detect')

        # Initialize member variables
        self._publishers_dict = {}
        self._bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('camera_name', 'camera1')
        self.declare_parameter('camera_topic', '/camera1/image_raw')
        self.declare_parameter('tag_size', 0.100)
        self.declare_parameter('tag_family', 'tag25h9')
        self.declare_parameter('calib_file', os.path.expanduser('~/Desktop/ost.yaml'))
        self.declare_parameter('get_debug_image', False)

        # Get parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.tag_size = self.get_parameter('tag_size').value
        self.calib_file = self.get_parameter('calib_file').value
        self.tag_family = self.get_parameter('tag_family').value
        self.get_debug_image = self.get_parameter('get_debug_image').value

        # Load camera calibration
        try:
            with open(self.calib_file, "r") as file:
                calib_data = yaml.safe_load(file)
            self.camera_matrix = np.array(calib_data["camera_matrix"]["data"]).reshape(3, 3)
            self.dist_coeffs = np.array(calib_data["distortion_coefficients"]["data"])
            self.get_logger().info("Camera calibration loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Error loading calibration file: {e}")
            raise

        # Initialize detector
        self._detector = Detector(
            families= self.tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25
        )

        # Create subscriber
        self._image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self._image_callback,
            1
        )

        # Initialize display window
        if self.get_debug_image:
            self._window_name = f'AprilTag Detections - {self.camera_name}'
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f'AprilTag detector initialized for {self.camera_name}')

    def _get_publisher(self, tag_id):
        """Get or create a publisher for a tag ID"""
        try:
            # Validate _publishers as a dictionary
            if not isinstance(self._publishers_dict, dict):
                self.get_logger().error(f"_publishers is not a dictionary! Type: {type(self._publishers_dict)}. Resetting to an empty dictionary.")
                self._publishers_dict = {}  # Reset to an empty dictionary

            # Ensure tag_id is an integer
            tag_id = int(tag_id)

            # Check if tag_id already exists in _publishers
            if tag_id in self._publishers_dict:
                return self._publishers_dict[tag_id]

            # Create and store a new publisher
            topic = f'/tag{tag_id}/{self.camera_name}'
            self._publishers_dict[tag_id] = self.create_publisher(PoseStamped, topic, 10)
            self.get_logger().info(f'Created publisher for tag {tag_id}')
            return self._publishers_dict[tag_id]
        except Exception as e:
            self.get_logger().error(f"Error creating/getting publisher for tag_id {tag_id}: {str(e)}")
            raise




    def _image_callback(self, msg):
        print("Image received")
        """Process incoming image messages"""
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            detections = self._detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[
                    self.camera_matrix[0, 0],
                    self.camera_matrix[1, 1],
                    self.camera_matrix[0, 2],
                    self.camera_matrix[1, 2]
                ],
                tag_size=self.tag_size
            )

            for detection in detections:
                print("Tag ID:", detection.tag_id)
                print("Tag Center:", detection.center)
                print("Tag Corners:", detection.corners)
                print("Tag Pose:", detection.pose_t, detection.pose_R)
            
                # Create and publish pose message
                if detection.pose_t is not None and detection.pose_R is not None:
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.header.frame_id = f"{self.camera_name}_optical_frame"

                    # Position
                    pose_msg.pose.position.x = float(detection.pose_t[0])
                    pose_msg.pose.position.y = float(detection.pose_t[1])
                    pose_msg.pose.position.z = float(detection.pose_t[2])

                    # Orientation (rotation matrix to quaternion)
                    r = detection.pose_R
                    # Convert rotation matrix to quaternion using SciPy
                    rotation = R.from_matrix(r)
                    qx, qy, qz, qw = rotation.as_quat()  # SciPy gives quaternion in [qx, qy, qz, qw]
                    pose_msg.pose.orientation.w = float(qw)
                    pose_msg.pose.orientation.x = float(qx)
                    pose_msg.pose.orientation.y = float(qy)
                    pose_msg.pose.orientation.z = float(qz)

                    # Publish
                    tag_id = int(detection.tag_id)
                    publisher = self._get_publisher(tag_id)
                    publisher.publish(pose_msg)
                print( "Pose:", detection.pose_t, detection.pose_R)
                if self.get_debug_image:
                    # Draw detection on the image
                    corners = np.array(detection.corners, dtype=np.int32)
                    cv2.polylines(cv_image, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                    center = tuple(map(int, detection.center))
                    cv2.putText(cv_image, f"ID: {detection.tag_id}", (center[0] - 20, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if detection.pose_t is not None:
                        dist = np.linalg.norm(detection.pose_t)
                        cv2.putText(cv_image, f"Dist: {dist:.2f}m", (center[0] - 20, center[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show image
            if self.get_debug_image:
                cv2.imshow(self._window_name, cv_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')

    def __del__(self):
        """Cleanup"""
        try:
            cv2.destroyWindow(self._window_name)
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AprilTagDetectorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
