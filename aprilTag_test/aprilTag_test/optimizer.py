#!/usr/bin/env python3

import numpy as np
import networkx as nx
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional, Set
import time
from dataclasses import dataclass
from .transform_types import Transform, Observation

@dataclass
class OptimizationStats:
    """Statistics from optimization run"""
    num_cameras: int
    num_tags: int
    num_observations: int
    cost_initial: float
    cost_final: float
    success: bool
    runtime: float

class GlobalOptimizer:
    def __init__(self, static_tag_threshold: int = 20):
        self.observations: Dict[Tuple[str, str], List[Observation]] = {}
        self.graph = nx.Graph()
        self.static_tag_threshold = static_tag_threshold
        self.last_optimization = None
        
    def add_observation(self, camera: str, tag: str, obs: Observation):
        """Add new observation to the system"""
        # Store observation
        key = (camera, tag)
        if key not in self.observations:
            self.observations[key] = []
        self.observations[key].append(obs)
        
        # Update graph
        if not self.graph.has_node(camera):
            self.graph.add_node(camera, node_type='camera')
        if not self.graph.has_node(tag):
            self.graph.add_node(tag, node_type='tag')
        self.graph.add_edge(camera, tag)
        
        # Clean old observations
        current_time = time.time()
        self.observations[key] = [
            o for o in self.observations[key]
            if current_time - o.timestamp < 1.0
        ]
        if not self.observations[key]:
            del self.observations[key]

    def find_transform_chain(self, start: str, end: str = 'tag0') -> List[Transform]:
        """Find chain of transforms between nodes"""
        try:
            path = nx.shortest_path(self.graph, start, end)
            transforms = []
            
            for i in range(len(path) - 1):
                node1, node2 = path[i:i+2]
                key = (node1, node2)
                rev_key = (node2, node1)
                
                if key in self.observations and self.observations[key]:
                    obs = max(self.observations[key], key=lambda x: x.timestamp)
                    transforms.append(Transform(
                        translation=obs.translation,
                        rotation=obs.rotation,
                        covariance=obs.covariance,
                        timestamp=obs.timestamp
                    ))
                elif rev_key in self.observations and self.observations[rev_key]:
                    obs = max(self.observations[rev_key], key=lambda x: x.timestamp)
                    transforms.append(Transform(
                        translation=obs.translation,
                        rotation=obs.rotation,
                        covariance=obs.covariance,
                        timestamp=obs.timestamp
                    ).inverse())
                else:
                    return []
            
            return transforms
            
        except nx.NetworkXNoPath:
            return []

    def optimize_transforms(self) -> Dict[str, Transform]:
        """Optimize transforms globally"""
        cameras = [n for n, d in self.graph.nodes(data=True) 
                  if d.get('node_type') == 'camera']
        static_tags = [f'tag{i}' for i in range(self.static_tag_threshold)
                      if self.graph.has_node(f'tag{i}')]
        
        if not cameras or not static_tags:
            return {}
            
        def _normalize_quaternion(quat):
            norm = np.linalg.norm(quat)
            if norm < 1e-10:
                return np.array([1.0, 0.0, 0.0, 0.0])
            return quat / norm
        
        def objective(params):
            errors = []
            idx = 0
            camera_poses = {}
            tag_poses = {}
            
            # Extract camera poses
            for cam in cameras:
                pos = params[idx:idx+3]
                quat = _normalize_quaternion(params[idx+3:idx+7])
                camera_poses[cam] = (pos, quat)
                idx += 7
            
            # Extract tag poses
            for tag in static_tags:
                if tag != 'tag0':
                    pos = params[idx:idx+3]
                    quat = _normalize_quaternion(params[idx+3:idx+7])
                    tag_poses[tag] = (pos, quat)
                    idx += 7
            
            # Compute errors
            for (camera, tag), observations in self.observations.items():
                if camera in camera_poses and tag in static_tags:
                    cam_pos, cam_quat = camera_poses[camera]
                    cam_rot = Rotation.from_quat([cam_quat[1], cam_quat[2], 
                                                cam_quat[3], cam_quat[0]]).as_matrix()
                    
                    if tag == 'tag0':
                        tag_pos = np.zeros(3)
                        tag_rot = np.eye(3)
                    else:
                        tag_pos, tag_quat = tag_poses[tag]
                        tag_rot = Rotation.from_quat([tag_quat[1], tag_quat[2],
                                                    tag_quat[3], tag_quat[0]]).as_matrix()
                    
                    for obs in observations:
                        pred_trans = np.dot(cam_rot.T, tag_pos - cam_pos)
                        pred_rot = np.dot(cam_rot.T, tag_rot)
                        
                        trans_error = np.linalg.norm(pred_trans - obs.translation)
                        rot_error = np.linalg.norm(pred_rot - obs.rotation, ord='fro')
                        
                        conf = 1.0 / np.sqrt(np.linalg.det(obs.covariance[:3, :3]) + 1e-10)
                        errors.extend([trans_error * conf, rot_error * conf])
            
            return np.array(errors)
        
        # Initial parameters
        n_params = 7 * (len(cameras) + len(static_tags) - 1)  # -1 for tag0
        initial_params = np.zeros(n_params)
        
        # Set initial quaternions to identity
        for i in range((len(cameras) + len(static_tags) - 1)):
            initial_params[i * 7 + 3] = 1.0
        
        # Optimize
        start_time = time.time()
        initial_cost = np.sum(objective(initial_params)**2)
        
        result = least_squares(
            objective,
            initial_params,
            method='dogbox',
            loss='soft_l1',
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=200
        )
        
        optimized_transforms = {}
        
        if result.success:
            idx = 0
            # Extract camera transforms
            for cam in cameras:
                pos = result.x[idx:idx+3]
                quat = _normalize_quaternion(result.x[idx+3:idx+7])
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                
                J = result.jac[idx:idx+7, :]
                cov = np.linalg.pinv(np.dot(J.T, J) + np.eye(J.shape[1]) * 1e-10)
                
                optimized_transforms[cam] = Transform(
                    translation=pos,
                    rotation=rot,
                    covariance=cov,
                    timestamp=time.time()
                )
                idx += 7
            
            # Extract tag transforms
            for tag in static_tags:
                if tag != 'tag0':
                    pos = result.x[idx:idx+3]
                    quat = _normalize_quaternion(result.x[idx+3:idx+7])
                    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                    
                    J = result.jac[idx:idx+7, :]
                    cov = np.linalg.pinv(np.dot(J.T, J) + np.eye(J.shape[1]) * 1e-10)
                    
                    optimized_transforms[tag] = Transform(
                        translation=pos,
                        rotation=rot,
                        covariance=cov,
                        timestamp=time.time()
                    )
                    idx += 7
            
            # Store optimization stats
            self.last_optimization = OptimizationStats(
                num_cameras=len(cameras),
                num_tags=len(static_tags),
                num_observations=sum(len(obs) for obs in self.observations.values()),
                cost_initial=initial_cost,
                cost_final=np.sum(result.fun**2),
                success=True,
                runtime=time.time() - start_time
            )
        
        return optimized_transforms

    def get_optimization_stats(self) -> Optional[OptimizationStats]:
        return self.last_optimization