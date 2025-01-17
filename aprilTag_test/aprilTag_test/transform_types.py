#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from transforms3d.quaternions import mat2quat, quat2mat
from typing import List, Optional
import time

@dataclass
class Observation:
    """Raw observation from a camera"""
    translation: np.ndarray
    rotation: np.ndarray
    covariance: np.ndarray
    timestamp: float

@dataclass
class Transform:
    """Transform with uncertainty"""
    translation: np.ndarray
    rotation: np.ndarray
    covariance: np.ndarray
    timestamp: float

    @property
    def confidence(self) -> float:
        """Compute confidence from covariance with safety checks"""
        try:
            pos_cov = self.covariance[:3, :3]
            det = np.linalg.det(pos_cov)
            
            # Handle invalid determinants
            if det <= 0:
                return 0.0
                
            # Bound the confidence between 0 and 1
            conf = 1.0 / (1.0 + np.sqrt(det))
            return min(max(conf, 0.0), 1.0)
            
        except Exception:
            return 0.0  # Return zero confidence if calculation fails

    def inverse(self) -> 'Transform':
        """Compute inverse transform"""
        inv_rot = self.rotation.T
        inv_trans = -np.dot(inv_rot, self.translation)
        
        J = np.zeros((6, 6))
        J[:3, :3] = -inv_rot
        J[3:, 3:] = inv_rot
        inv_cov = np.dot(np.dot(J, self.covariance), J.T)
        
        return Transform(inv_trans, inv_rot, inv_cov, self.timestamp)

    @staticmethod
    def compose(t1: 'Transform', t2: 'Transform') -> 'Transform':
        """Compose two transforms"""
        rot = np.dot(t1.rotation, t2.rotation)
        trans = np.dot(t1.rotation, t2.translation) + t1.translation
        
        J1 = np.zeros((6, 6))
        J1[:3, :3] = t1.rotation
        J1[3:, 3:] = np.eye(3)
        
        J2 = np.zeros((6, 6))
        J2[:3, :3] = np.eye(3)
        J2[:3, 3:] = -np.dot(t1.rotation, Transform.skew(t2.translation))
        J2[3:, 3:] = t1.rotation
        
        cov = np.dot(np.dot(J1, t2.covariance), J1.T) + \
              np.dot(np.dot(J2, t1.covariance), J2.T)
              
        return Transform(trans, rot, cov, max(t1.timestamp, t2.timestamp))

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def average_transforms(transforms: List['Transform'], 
                         weights: Optional[List[float]] = None) -> 'Transform':
        """Weighted average of transforms"""
        if not transforms:
            raise ValueError("No transforms to average")
            
        if weights is None:
            weights = [1.0/len(transforms)] * len(transforms)
        weights = np.array(weights) / np.sum(weights)
        
        avg_trans = np.zeros(3)
        for t, w in zip(transforms, weights):
            avg_trans += w * t.translation
        
        quats = [mat2quat(t.rotation) for t in transforms]
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        avg_quat = np.zeros(4)
        for q, w in zip(quats, weights):
            avg_quat += w * q
        avg_quat /= np.linalg.norm(avg_quat)
        
        avg_cov = np.zeros((6, 6))
        for t, w in zip(transforms, weights):
            avg_cov += w * t.covariance
            
        return Transform(
            translation=avg_trans,
            rotation=quat2mat(avg_quat),
            covariance=avg_cov,
            timestamp=max(t.timestamp for t in transforms)
        )