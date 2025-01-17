#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import time
from typing import Dict, List, Optional
import threading

from .transform_types import Transform, Observation
from .optimizer import GlobalOptimizer

class TransformManagerNode(Node):
    def __init__(self):
        super().__init__('transform_manager')
        
        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Initialize optimizer
        self.optimizer = GlobalOptimizer(static_tag_threshold=20)
        
        # Store robot tag observations
        self.robot_tag_observations: Dict[tuple, Transform] = {}
        
        # Lock for thread safety
        self.transform_lock = threading.Lock()
        
        # Parameters
        self.declare_parameter('camera_names', ['camera1', 'camera2'])
        self.declare_parameter('tf_publish_rate', 60.0)  # Hz
        self.declare_parameter('optimization_rate', 30.0)  # Hz
        
        self.camera_names = self.get_parameter('camera_names').value
        self.static_tag_threshold = 20
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Create timers
        publish_period = 1.0 / self.get_parameter('tf_publish_rate').value
        optimization_period = 1.0 / self.get_parameter('optimization_rate').value
        
        self.create_timer(publish_period, self.publish_transforms)
        self.create_timer(optimization_period, self.run_optimization)
        
        # Store optimized transforms
        self.optimized_transforms = {}
        
        self.get_logger().info('Transform manager initialized')

    def setup_subscribers(self):
        """Setup subscribers for all tags from all cameras"""
        self.pose_subs = []
        for camera in self.camera_names:
            for tag_id in range(50):  # Assuming max 50 tags
                topic = f'/tag{tag_id}/{camera}'
                sub = self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, c=camera, t=tag_id: self.pose_callback(msg, c, t),
                    10)
                self.pose_subs.append(sub)

    def pose_to_observation(self, msg: PoseStamped) -> Observation:
        """Convert PoseStamped to Observation with robust covariance"""
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
        
        # More robust distance-based confidence
        distance = np.linalg.norm(translation)
        min_variance = 1e-6  # Minimum variance to ensure positive definiteness
        
        # Scale variances with distance but ensure minimum value
        pos_variance = max(min_variance, (0.01 + 0.01 * distance)**2)
        rot_variance = max(min_variance, (0.01 + 0.02 * distance)**2)
        
        # Create positive definite covariance matrix
        covariance = np.diag([pos_variance] * 3 + [rot_variance] * 3)
        
        # Add small regularization term to ensure positive definiteness
        covariance += np.eye(6) * min_variance
        
        return Observation(
            translation=translation,
            rotation=rotation,
            covariance=covariance,
            timestamp=time.time()
        )

    def pose_callback(self, msg: PoseStamped, camera_name: str, tag_id: int):
        """Handle incoming tag poses with better error handling"""
        try:
            # Skip processing if tag_id is invalid
            if not (0 <= tag_id < 50):  # Assuming max 50 tags
                return
                
            observation = self.pose_to_observation(msg)
            tag_name = f'tag{tag_id}'
            
            with self.transform_lock:
                if tag_id < self.static_tag_threshold:
                    # Add to optimizer for static tags
                    self.optimizer.add_observation(camera_name, tag_name, observation)
                else:
                    # Handle robot tag
                    if camera_name in self.optimized_transforms:
                        camera_transform = self.optimized_transforms[camera_name]
                        
                        # Skip if camera transform has low confidence
                        if camera_transform.confidence < 0.1:
                            return
                            
                        tag_transform = Transform(
                            translation=observation.translation,
                            rotation=observation.rotation,
                            covariance=observation.covariance,
                            timestamp=observation.timestamp
                        )
                        
                        tag_in_tag0 = Transform.compose(camera_transform, tag_transform)
                        
                        # Only store if confidence is reasonable
                        if tag_in_tag0.confidence > 0.1:
                            self.robot_tag_observations[(camera_name, tag_id)] = tag_in_tag0
            
        except Exception as e:
            self.get_logger().error(f'Error in pose callback for tag {tag_id}: {str(e)}')


    def get_robot_tag_transform(self, tag_id: int) -> Optional[Transform]:
        """Compute robust transform for robot tag using multi-camera fusion"""
        current_time = time.time()
        recent_observations = []
        observation_weights = []
        max_age = 0.1  # 100ms window

        # Collect recent observations
        for (camera, tid), transform in self.robot_tag_observations.items():
            if tid == tag_id and (current_time - transform.timestamp) < max_age:
                if camera in self.optimized_transforms:
                    camera_confidence = self.optimized_transforms[camera].confidence
                    
                    # Calculate weights
                    distance = np.linalg.norm(transform.translation)
                    distance_weight = 1.0 / (1.0 + distance)
                    recency_weight = 1.0 / (current_time - transform.timestamp + 1e-6)
                    
                    # Combined weight
                    total_weight = distance_weight * recency_weight * camera_confidence
                    
                    recent_observations.append(transform)
                    observation_weights.append(total_weight)

        if not recent_observations:
            return None

        # Normalize weights
        total_weight = sum(observation_weights)
        if total_weight > 0:
            observation_weights = [w/total_weight for w in observation_weights]

        # Single observation case
        if len(recent_observations) == 1:
            return recent_observations[0]

        # Multi-camera fusion
        mean_pos = np.zeros(3)
        for obs, weight in zip(recent_observations, observation_weights):
            mean_pos += obs.translation * weight

        # Outlier rejection
        valid_obs = []
        valid_weights = []
        threshold = 0.1  # 10cm threshold
        
        for obs, weight in zip(recent_observations, observation_weights):
            if np.linalg.norm(obs.translation - mean_pos) < threshold:
                valid_obs.append(obs)
                valid_weights.append(weight)

        if not valid_obs:
            valid_obs = recent_observations
            valid_weights = observation_weights

        # Renormalize weights
        total_weight = sum(valid_weights)
        valid_weights = [w/total_weight for w in valid_weights]

        # Combine transforms
        return Transform.average_transforms(valid_obs, valid_weights)

    def run_optimization(self):
        """Run global optimization"""
        try:
            with self.transform_lock:
                self.optimized_transforms = self.optimizer.optimize_transforms()
        except Exception as e:
            self.get_logger().error(f'Error in optimization: {str(e)}')

    def create_tf_msg(self, parent: str, child: str, 
                     transform: Transform, stamp) -> TransformStamped:
        """Create TF2 message"""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child
        
        tf_msg.transform.translation.x = float(transform.translation[0])
        tf_msg.transform.translation.y = float(transform.translation[1])
        tf_msg.transform.translation.z = float(transform.translation[2])
        
        quat = mat2quat(transform.rotation)
        tf_msg.transform.rotation.w = float(quat[0])
        tf_msg.transform.rotation.x = float(quat[1])
        tf_msg.transform.rotation.y = float(quat[2])
        tf_msg.transform.rotation.z = float(quat[3])
        
        return tf_msg

    def publish_transforms(self):
        """Publish transforms to TF2"""
        try:
            current_time = self.get_clock().now().to_msg()
            
            with self.transform_lock:
                # Publish static transforms
                for name, transform in self.optimized_transforms.items():
                    if transform.confidence > 0.1:
                        tf_msg = self.create_tf_msg(
                            'tag0', name, transform, current_time)
                        self.tf_broadcaster.sendTransform(tf_msg)
                
                # Publish robot tag transforms
                seen_tags = set()
                for (_, tag_id) in self.robot_tag_observations.keys():
                    if tag_id not in seen_tags:
                        seen_tags.add(tag_id)
                        transform = self.get_robot_tag_transform(tag_id)
                        
                        if transform and transform.confidence > 0.1:
                            tf_msg = self.create_tf_msg(
                                'tag0', f'tag{tag_id}', transform, current_time)
                            self.tf_broadcaster.sendTransform(tf_msg)
                            
                            # Debug multi-camera info
                            num_cameras = sum(1 for k in self.robot_tag_observations.keys() 
                                           if k[1] == tag_id)
                            if num_cameras > 1:
                                self.get_logger().debug(
                                    f'Robot tag{tag_id} seen by {num_cameras} cameras, '
                                    f'confidence: {transform.confidence:.3f}'
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