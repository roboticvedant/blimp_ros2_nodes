#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import networkx as nx
from transforms3d.quaternions import mat2quat, quat2mat
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Set
import time
from scipy.optimize import least_squares

@dataclass
class Transform:
    translation: np.ndarray
    rotation: np.ndarray
    confidence: float
    timestamp: float

class TransformManagerNode(Node):
    def __init__(self):
        super().__init__('transform_manager')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Graph for static transforms (cameras and ground tags)
        self.static_graph = nx.Graph()
        self.static_graph.add_node('tag0')
        
        # Store transforms and observations
        self.transforms_to_tag0: Dict[str, Transform] = {}  # For cameras and static tags
        self.robot_transforms: Dict[str, Transform] = {}    # For robot tags (>20)
        self.camera_observations: Dict[Tuple[str, str], Transform] = {}
        
        # Track which tags each camera sees
        self.static_tags_per_camera: Dict[str, Set[str]] = {}
        self.robot_tags_per_camera: Dict[str, Set[str]] = {}
        
        # Parameters
        self.declare_parameter('camera_names', ['camera1', 'camera2'])
        self.camera_names = self.get_parameter('camera_names').value
        self.optimization_window = 1.0  # seconds
        
        # Constants
        self.STATIC_TAG_THRESHOLD = 20  # Tags 0-19 are static ground tags
        
        # Initialize cameras
        for camera in self.camera_names:
            self.static_graph.add_node(camera)
            self.static_tags_per_camera[camera] = set()
            self.robot_tags_per_camera[camera] = set()
            self.create_tag_subscribers(camera)
        
        self.create_timer(0.1, self.publish_transforms)
        self.get_logger().info('Transform manager initialized with static/dynamic tag handling')

    def create_tag_subscribers(self, camera_name: str):
        """Create subscribers for tag poses from a camera"""
        # Subscribe to all possible tags (both static and dynamic)
        for tag_id in range(35):  # Assuming max 35 tags total
            topic = f'/tag{tag_id}/{camera_name}'
            sub = self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, c=camera_name, t=tag_id: self.pose_callback(msg, c, t),
                10
            )

    def is_static_tag(self, tag_id: int) -> bool:
        """Check if a tag is a static ground tag"""
        return tag_id < self.STATIC_TAG_THRESHOLD

    def optimize_camera_position(self, camera_name: str):
        """Optimize camera position using all visible static tags"""
        static_tags = self.static_tags_per_camera[camera_name]
        if not static_tags:
            return None

        def objective(x):
            # x = [tx, ty, tz, rx, ry, rz] (camera position and orientation)
            cam_trans = x[:3]
            cam_rot = self.rodrigues_to_mat(x[3:])
            
            errors = []
            for tag in static_tags:
                if tag in self.transforms_to_tag0:  # We know the tag's position
                    obs = self.camera_observations.get((camera_name, tag))
                    if obs:
                        # Predict where we should see the tag
                        tag_trans = self.transforms_to_tag0[tag].translation
                        tag_rot = self.transforms_to_tag0[tag].rotation
                        
                        # Transform from camera to tag
                        pred_trans = np.dot(cam_rot.T, (tag_trans - cam_trans))
                        pred_rot = np.dot(cam_rot.T, tag_rot)
                        
                        # Compare with actual observation
                        trans_error = np.linalg.norm(pred_trans - obs.translation)
                        rot_error = np.linalg.norm(pred_rot - obs.rotation)
                        errors.extend([trans_error, rot_error])
            
            return np.array(errors)

        # Initial guess from current transform if available
        initial_guess = np.zeros(6)
        if camera_name in self.transforms_to_tag0:
            initial_guess[:3] = self.transforms_to_tag0[camera_name].translation
            initial_guess[3:] = self.mat_to_rodrigues(self.transforms_to_tag0[camera_name].rotation)

        result = least_squares(objective, initial_guess)
        if result.success:
            return Transform(
                translation=result.x[:3],
                rotation=self.rodrigues_to_mat(result.x[3:]),
                confidence=1.0 / (1.0 + result.cost),
                timestamp=time.time()
            )
        return None

    def pose_callback(self, msg: PoseStamped, camera_name: str, tag_id: int):
        try:
            tag_name = f'tag{tag_id}'
            
            # Extract transform
            translation = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            rotation = quat2mat([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ])
            
            # Store observation
            confidence = 1.0 / (1.0 + np.linalg.norm(translation))
            self.camera_observations[(camera_name, tag_name)] = Transform(
                translation=translation,
                rotation=rotation,
                confidence=confidence,
                timestamp=time.time()
            )

            # Handle static ground tags (0-19)
            if self.is_static_tag(tag_id):
                self.static_tags_per_camera[camera_name].add(tag_name)
                self.static_graph.add_edge(camera_name, tag_name)
                
                # If this is tag0, update camera transform directly
                if tag_id == 0:
                    inv_rot = rotation.T
                    inv_trans = -np.dot(inv_rot, translation)
                    self.transforms_to_tag0[camera_name] = Transform(
                        translation=inv_trans,
                        rotation=inv_rot,
                        confidence=confidence,
                        timestamp=time.time()
                    )
                    
                # Optimize camera position using all visible static tags
                optimized_camera = self.optimize_camera_position(camera_name)
                if optimized_camera:
                    self.transforms_to_tag0[camera_name] = optimized_camera
                    
                # Update positions of all robot tags seen by this camera
                self.update_robot_tags(camera_name)
                
            # Handle robot tags (20+)
            else:
                self.robot_tags_per_camera[camera_name].add(tag_name)
                if camera_name in self.transforms_to_tag0:
                    # Transform robot tag position to tag0 frame
                    cam_transform = self.transforms_to_tag0[camera_name]
                    tag_trans = np.dot(cam_transform.rotation, translation) + cam_transform.translation
                    tag_rot = np.dot(cam_transform.rotation, rotation)
                    
                    self.robot_transforms[tag_name] = Transform(
                        translation=tag_trans,
                        rotation=tag_rot,
                        confidence=confidence * cam_transform.confidence,
                        timestamp=time.time()
                    )
            
        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {str(e)}')

    def update_robot_tags(self, camera_name: str):
        """Update positions of all robot tags seen by a camera"""
        if camera_name not in self.transforms_to_tag0:
            return
            
        cam_transform = self.transforms_to_tag0[camera_name]
        
        for tag_name in self.robot_tags_per_camera[camera_name]:
            obs = self.camera_observations.get((camera_name, tag_name))
            if obs:
                # Transform to tag0 frame
                tag_trans = np.dot(cam_transform.rotation, obs.translation) + cam_transform.translation
                tag_rot = np.dot(cam_transform.rotation, obs.rotation)
                
                self.robot_transforms[tag_name] = Transform(
                    translation=tag_trans,
                    rotation=tag_rot,
                    confidence=obs.confidence * cam_transform.confidence,
                    timestamp=time.time()
                )

    def publish_transforms(self):
        try:
            current_time = self.get_clock().now().to_msg()
            
            # 1. Publish static ground tag transforms (0-19)
            for camera_name in self.camera_names:
                for tag_name in self.static_tags_per_camera[camera_name]:
                    if (camera_name, tag_name) in self.camera_observations:
                        obs = self.camera_observations[(camera_name, tag_name)]
                        if camera_name in self.transforms_to_tag0:
                            # Transform from tag0 to this tag through camera
                            cam_to_tag0 = self.transforms_to_tag0[camera_name]
                            
                            # Calculate tag position in tag0 frame
                            tag_trans = np.dot(cam_to_tag0.rotation, obs.translation) + cam_to_tag0.translation
                            tag_rot = np.dot(cam_to_tag0.rotation, obs.rotation)
                            
                            # Create and publish transform
                            tf_msg = TransformStamped()
                            tf_msg.header.stamp = current_time
                            tf_msg.header.frame_id = 'tag0'
                            tf_msg.child_frame_id = tag_name
                            
                            # Set translation
                            tf_msg.transform.translation.x = float(tag_trans[0])
                            tf_msg.transform.translation.y = float(tag_trans[1])
                            tf_msg.transform.translation.z = float(tag_trans[2])
                            
                            # Set rotation
                            quat = mat2quat(tag_rot)
                            tf_msg.transform.rotation.w = float(quat[0])
                            tf_msg.transform.rotation.x = float(quat[1])
                            tf_msg.transform.rotation.y = float(quat[2])
                            tf_msg.transform.rotation.z = float(quat[3])
                            
                            self.tf_broadcaster.sendTransform(tf_msg)
            
            # 2. Publish camera transforms
            for camera_name, transform in self.transforms_to_tag0.items():
                if transform.confidence > 0.1:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = current_time
                    tf_msg.header.frame_id = 'tag0'
                    tf_msg.child_frame_id = camera_name
                    
                    tf_msg.transform.translation.x = float(transform.translation[0])
                    tf_msg.transform.translation.y = float(transform.translation[1])
                    tf_msg.transform.translation.z = float(transform.translation[2])
                    
                    quat = mat2quat(transform.rotation)
                    tf_msg.transform.rotation.w = float(quat[0])
                    tf_msg.transform.rotation.x = float(quat[1])
                    tf_msg.transform.rotation.y = float(quat[2])
                    tf_msg.transform.rotation.z = float(quat[3])
                    
                    self.tf_broadcaster.sendTransform(tf_msg)

            # 3. Publish robot tag transforms (20+)
            for tag_name, transform in self.robot_transforms.items():
                if transform.confidence > 0.1:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = current_time
                    tf_msg.header.frame_id = 'tag0'
                    tf_msg.child_frame_id = tag_name
                    
                    tf_msg.transform.translation.x = float(transform.translation[0])
                    tf_msg.transform.translation.y = float(transform.translation[1])
                    tf_msg.transform.translation.z = float(transform.translation[2])
                    
                    quat = mat2quat(transform.rotation)
                    tf_msg.transform.rotation.w = float(quat[0])
                    tf_msg.transform.rotation.x = float(quat[1])
                    tf_msg.transform.rotation.y = float(quat[2])
                    tf_msg.transform.rotation.z = float(quat[3])
                    
                    self.tf_broadcaster.sendTransform(tf_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing transforms: {str(e)}')

    @staticmethod
    def rodrigues_to_mat(rvec):
        """Convert rotation vector to matrix"""
        theta = np.linalg.norm(rvec)
        if theta < 1e-6:
            return np.eye(3)
        
        r = rvec / theta
        R = np.array([[0, -r[2], r[1]],
                     [r[2], 0, -r[0]],
                     [-r[1], r[0], 0]])
        return np.eye(3) + np.sin(theta) * R + (1 - np.cos(theta)) * np.dot(R, R)

    @staticmethod
    def mat_to_rodrigues(rmat):
        """Convert rotation matrix to vector"""
        A = (rmat - rmat.T) / 2
        rho = np.array([A[2, 1], A[0, 2], A[1, 0]])
        s = np.linalg.norm(rho)
        c = (np.trace(rmat) - 1) / 2
        
        if s < 1e-6:
            return np.zeros(3)
            
        theta = np.arctan2(s, c)
        return theta * (rho / s)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TransformManagerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()