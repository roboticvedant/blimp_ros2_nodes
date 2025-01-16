#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import math

class TransformManagerNode(Node):
    def __init__(self):
        super().__init__('transform_manager')
        
        # Initialize transform broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Store camera transforms relative to tag0
        self.camera_transforms = {}
        self.tag_transforms = {}
        
        # Create subscriptions for cameras
        self.declare_parameter('camera_names', ['camera1', 'camera2'])
        self.camera_names = self.get_parameter('camera_names').value
        
        # Subscribe to tag0 poses from each camera to establish base transforms
        for camera in self.camera_names:
            self.create_subscription(
                PoseStamped,
                f'/tag0/{camera}',
                lambda msg, camera=camera: self.tag0_callback(msg, camera),
                10
            )
        
        # Subscribe to other tag poses
        self.tag_subs = {}  # Store subscribers for each tag
        
        # Timer to publish the complete TF tree
        self.create_timer(0.033, self.publish_tf_tree)  # 30Hz
        
        self.get_logger().info('Transform manager initialized')

    def tag0_callback(self, msg, camera_name):
        """Handle poses of tag0 from different cameras to establish base reference frame"""
        try:
            # Create transform from camera to tag0
            transform = TransformStamped()
            transform.header = msg.header
            transform.header.frame_id = f"{camera_name}_optical_frame"
            transform.child_frame_id = "tag0"
            
            # Copy pose to transform
            transform.transform.translation.x = msg.pose.position.x
            transform.transform.translation.y = msg.pose.position.y
            transform.transform.translation.z = msg.pose.position.z
            transform.transform.rotation = msg.pose.orientation
            
            # Store the transform
            self.camera_transforms[camera_name] = transform
            
            # Subscribe to other tags for this camera if not already subscribed
            self.subscribe_to_other_tags(camera_name)
            
        except Exception as e:
            self.get_logger().error(f'Error processing tag0 pose from {camera_name}: {str(e)}')

    def subscribe_to_other_tags(self, camera_name):
        """Create subscriptions for other tags from this camera"""
        for tag_id in range(1, 10):  # Adjust range based on expected number of tags
            topic = f'/tag{tag_id}/{camera_name}'
            if (tag_id, camera_name) not in self.tag_subs:
                sub = self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, tag_id=tag_id, camera=camera_name: self.tag_callback(msg, tag_id, camera),
                    10
                )
                self.tag_subs[(tag_id, camera_name)] = sub

    def tag_callback(self, msg, tag_id, camera_name):
        """Handle poses of other tags"""
        try:
            if camera_name in self.camera_transforms:
                # Store the tag's pose relative to the camera
                key = (tag_id, camera_name)
                if key not in self.tag_transforms:
                    self.tag_transforms[key] = {}
                
                self.tag_transforms[key] = {
                    'header': msg.header,
                    'translation': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                    'rotation': [msg.pose.orientation.w, msg.pose.orientation.x, 
                               msg.pose.orientation.y, msg.pose.orientation.z]
                }
                
        except Exception as e:
            self.get_logger().error(f'Error processing tag {tag_id} pose from {camera_name}: {str(e)}')

    def compute_tag0_to_tag_transform(self, tag_id, camera_name):
        """Compute transform from tag0 to another tag using camera-tag0 and camera-tag transforms"""
        try:
            if camera_name not in self.camera_transforms:
                return None
                
            key = (tag_id, camera_name)
            if key not in self.tag_transforms:
                return None
            
            # Get camera to tag0 transform (inverse it)
            cam_to_tag0 = self.camera_transforms[camera_name]
            tag0_rot = quat2mat([cam_to_tag0.transform.rotation.w,
                               cam_to_tag0.transform.rotation.x,
                               cam_to_tag0.transform.rotation.y,
                               cam_to_tag0.transform.rotation.z])
            tag0_trans = np.array([cam_to_tag0.transform.translation.x,
                                 cam_to_tag0.transform.translation.y,
                                 cam_to_tag0.transform.translation.z])
            
            # Get camera to tag transform
            tag_data = self.tag_transforms[key]
            tag_rot = quat2mat(tag_data['rotation'])
            tag_trans = np.array(tag_data['translation'])
            
            # Compute tag0 to tag transform
            relative_rot = np.dot(tag0_rot.T, tag_rot)
            relative_trans = np.dot(tag0_rot.T, tag_trans - tag0_trans)
            
            # Create transform message
            transform = TransformStamped()
            transform.header = tag_data['header']
            transform.header.frame_id = "tag0"
            transform.child_frame_id = f"tag{tag_id}"
            
            # Set translation
            transform.transform.translation.x = float(relative_trans[0])
            transform.transform.translation.y = float(relative_trans[1])
            transform.transform.translation.z = float(relative_trans[2])
            
            # Set rotation
            quat = mat2quat(relative_rot)
            transform.transform.rotation.w = float(quat[0])
            transform.transform.rotation.x = float(quat[1])
            transform.transform.rotation.y = float(quat[2])
            transform.transform.rotation.z = float(quat[3])
            
            return transform
            
        except Exception as e:
            self.get_logger().error(f'Error computing transform for tag {tag_id}: {str(e)}')
            return None

    def publish_tf_tree(self):
        """Publish the complete TF tree"""
        try:
            current_time = self.get_clock().now().to_msg()
            
            # Publish camera to tag0 transforms
            for camera_name, transform in self.camera_transforms.items():
                transform.header.stamp = current_time
                self.tf_broadcaster.sendTransform(transform)
            
            # Publish tag0 to other tag transforms
            for tag_id in range(1, 10):  # Adjust range based on expected number of tags
                best_transform = None
                best_camera = None
                
                # Find the best transform from available cameras
                for camera_name in self.camera_names:
                    transform = self.compute_tag0_to_tag_transform(tag_id, camera_name)
                    if transform is not None:
                        best_transform = transform
                        best_camera = camera_name
                        break  # Use the first available transform for now
                
                if best_transform is not None:
                    best_transform.header.stamp = current_time
                    self.tf_broadcaster.sendTransform(best_transform)
                    
        except Exception as e:
            self.get_logger().error(f'Error publishing TF tree: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TransformManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()