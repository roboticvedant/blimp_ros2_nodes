#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import networkx as nx
import time
from typing import Dict, Optional, List, Tuple

class Transform:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray, timestamp: float = None):
        self.translation = translation
        self.rotation = rotation
        self.timestamp = timestamp if timestamp else time.time()

    @staticmethod
    def from_pose(msg: PoseStamped) -> 'Transform':
        return Transform(
            translation=np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]),
            rotation=quat2mat([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ])
        )

    def inverse(self) -> 'Transform':
        inv_rot = self.rotation.T
        inv_trans = -np.dot(inv_rot, self.translation)
        return Transform(inv_trans, inv_rot, self.timestamp)

    def compose(self, other: 'Transform') -> 'Transform':
        final_rot = np.dot(self.rotation, other.rotation)
        final_trans = np.dot(self.rotation, other.translation) + self.translation
        return Transform(final_trans, final_rot, max(self.timestamp, other.timestamp))

    @staticmethod
    def average_transforms(transforms: List['Transform'], 
                         weights: Optional[List[float]] = None) -> 'Transform':
        if not transforms:
            raise ValueError("No transforms to average")
            
        if weights is None:
            weights = [1.0/len(transforms)] * len(transforms)
        weights = np.array(weights) / np.sum(weights)
        
        # Average translations
        avg_trans = np.zeros(3)
        for t, w in zip(transforms, weights):
            avg_trans += w * t.translation
            
        # Average rotations using quaternions
        quats = [mat2quat(t.rotation) for t in transforms]
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        avg_quat = np.zeros(4)
        for q, w in zip(quats, weights):
            avg_quat += w * q
        avg_quat /= np.linalg.norm(avg_quat)
        
        return Transform(avg_trans, quat2mat(avg_quat), max(t.timestamp for t in transforms))

class TransformManagerNode(Node):
    def __init__(self):
        super().__init__('transform_manager')
        
        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create transform graph
        self.transform_graph = nx.Graph()
        self.transform_edges: Dict[Tuple[str, str], Transform] = {}
        
        # Store camera and tag transforms
        self.static_tags: Dict[str, Transform] = {}  # tag -> tag0 transforms
        self.robot_observations: Dict[Tuple[str, int], Transform] = {}  # (camera, tag) -> transform
        
        # Parameters
        self.declare_parameter('camera_names', ['camera1', 'camera2'])
        self.declare_parameter('publish_rate', 30.0)
        
        self.camera_names = self.get_parameter('camera_names').value
        self.static_tag_threshold = 20
        
        # Camera correction for overhead mounting
        self.camera_correction = np.array([
            [ 0,  0,  1],  # Forward
            [-1,  0,  0],  # Left
            [ 0, -1,  0]   # Up
        ])
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Timer for publishing transforms
        publish_period = 1.0 / self.get_parameter('publish_rate').value
        self.create_timer(publish_period, self.publish_transforms)
        
        self.get_logger().info('Transform manager initialized')

    def setup_subscribers(self):
        self.tag_subs = []
        for camera in self.camera_names:
            for tag_id in range(50):  # Assuming max 50 tags
                topic = f'/tag{tag_id}/{camera}'
                sub = self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, c=camera, t=tag_id: self.tag_callback(msg, c, t),
                    10)
                self.tag_subs.append(sub)

    def add_transform_edge(self, frame1: str, frame2: str, transform: Transform):
        """Add transform to graph"""
        self.transform_graph.add_edge(frame1, frame2)
        self.transform_edges[(frame1, frame2)] = transform
        # Also store inverse transform
        self.transform_edges[(frame2, frame1)] = transform.inverse()

    def find_transform(self, source: str, target: str) -> Optional[Transform]:
        """Find transform between frames using graph"""
        try:
            if not (self.transform_graph.has_node(source) and 
                   self.transform_graph.has_node(target)):
                return None

            # Find shortest path
            path = nx.shortest_path(self.transform_graph, source, target)
            if len(path) < 2:
                return None

            # Compose transforms along path
            result = None
            for i in range(len(path) - 1):
                start, end = path[i:i+2]
                transform = self.transform_edges.get((start, end))
                if transform is None:
                    return None
                    
                if result is None:
                    result = transform
                else:
                    result = result.compose(transform)
            
            return result
            
        except nx.NetworkXNoPath:
            return None

    def tag_callback(self, msg: PoseStamped, camera_name: str, tag_id: int):
        try:
            # Convert message to transform
            camera_tf = Transform.from_pose(msg)
            
            # Apply camera correction
            camera_tf.rotation = np.dot(camera_tf.rotation, self.camera_correction)
            
            frame1 = camera_name
            frame2 = f'tag{tag_id}'
            
            if tag_id < self.static_tag_threshold:  # Static tag
                # Add to transform graph
                self.add_transform_edge(frame1, frame2, camera_tf)
                
                # Try to update static tag transform relative to tag0
                if frame2 != 'tag0':
                    tag_to_ref = self.find_transform(frame2, 'tag0')
                    if tag_to_ref is not None:
                        self.static_tags[frame2] = tag_to_ref
                
            else:  # Robot tag
                # First find camera's transform to tag0
                camera_to_ref = self.find_transform(camera_name, 'tag0')
                if camera_to_ref is not None:
                    # Store robot tag observation
                    self.robot_observations[(camera_name, tag_id)] = camera_tf
                    
        except Exception as e:
            self.get_logger().error(f'Error in tag callback: {str(e)}')

    def get_robot_tag_transform(self, tag_id: int) -> Optional[Transform]:
        """Compute average transform for robot tag"""
        current_time = time.time()
        valid_observations = []
        
        # Collect recent observations
        for (camera, tid), transform in self.robot_observations.items():
            if (tid == tag_id and 
                current_time - transform.timestamp < 0.1):  # Last 100ms
                
                # Get camera's transform to tag0
                camera_to_ref = self.find_transform(camera, 'tag0')
                if camera_to_ref is not None:
                    # Transform robot tag to tag0 frame
                    tag_in_ref = camera_to_ref.compose(transform)
                    valid_observations.append(tag_in_ref)
        
        if not valid_observations:
            return None
            
        # Average all valid observations
        if len(valid_observations) == 1:
            return valid_observations[0]
            
        # Weight by recency
        timestamps = np.array([obs.timestamp for obs in valid_observations])
        max_time = np.max(timestamps)
        weights = 1.0 / (1.0 + (max_time - timestamps))
        
        return Transform.average_transforms(valid_observations, weights)

    def create_transform_msg(self, parent: str, child: str, 
                           transform: Transform, current_time) -> TransformStamped:
        """Create TransformStamped message"""
        msg = TransformStamped()
        msg.header.stamp = current_time
        msg.header.frame_id = parent
        msg.child_frame_id = child
        
        msg.transform.translation.x = float(transform.translation[0])
        msg.transform.translation.y = float(transform.translation[1])
        msg.transform.translation.z = float(transform.translation[2])
        
        quat = mat2quat(transform.rotation)
        msg.transform.rotation.w = float(quat[0])
        msg.transform.rotation.x = float(quat[1])
        msg.transform.rotation.y = float(quat[2])
        msg.transform.rotation.z = float(quat[3])
        
        return msg

    def publish_transforms(self):
        try:
            current_time = self.get_clock().now().to_msg()
            
            # Publish camera transforms
            for camera in self.camera_names:
                transform = self.find_transform(camera, 'tag0')
                if transform is not None:
                    msg = self.create_transform_msg('tag0', camera, transform, current_time)
                    self.tf_broadcaster.sendTransform(msg)
            
            # Publish static tag transforms
            for tag_name, transform in self.static_tags.items():
                msg = self.create_transform_msg('tag0', tag_name, transform, current_time)
                self.tf_broadcaster.sendTransform(msg)
            
            # Publish robot tag transforms
            seen_tags = set()
            for (_, tag_id) in self.robot_observations.keys():
                if tag_id not in seen_tags and tag_id >= self.static_tag_threshold:
                    seen_tags.add(tag_id)
                    transform = self.get_robot_tag_transform(tag_id)
                    if transform is not None:
                        msg = self.create_transform_msg('tag0', f'tag{tag_id}', 
                                                      transform, current_time)
                        self.tf_broadcaster.sendTransform(msg)
                        
                        # Debug output for multi-camera observations
                        num_observers = sum(1 for k in self.robot_observations.keys() 
                                         if k[1] == tag_id)
                        if num_observers > 1:
                            self.get_logger().info(
                                f'Robot tag{tag_id} averaged from {num_observers} cameras'
                            )
            
        except Exception as e:
            self.get_logger().error(f'Error publishing transforms: {str(e)}')

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