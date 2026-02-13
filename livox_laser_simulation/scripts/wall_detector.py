#!/usr/bin/env python3

import numpy as np
import open3d as o3d


class WallDetector:
    """Detects walls from point cloud data using RANSAC plane segmentation"""
    
    def __init__(self, voxel_size=0.05, distance_threshold=0.1, min_points=200, 
                 vertical_threshold=0.2, ransac_n=3, num_iterations=1000):
        """
        Initialize wall detector with parameters
        
        Args:
            voxel_size: Voxel size for downsampling (m)
            distance_threshold: RANSAC distance threshold (m)
            min_points: Minimum points for a valid plane
            vertical_threshold: Maximum vertical component of normal for walls
            ransac_n: Number of points for RANSAC
            num_iterations: RANSAC iterations
        """
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.min_points = min_points
        self.vertical_threshold = vertical_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
    
    def detect(self, points):
        """
        Detect walls using iterative RANSAC plane segmentation
        
        Args:
            points: Nx3 numpy array of 3D points
            
        Returns:
            List of wall dictionaries with keys:
                - 'plane_model': [a, b, c, d] plane equation
                - 'points': Nx3 numpy array of points on the wall
                - 'num_points': Number of points
                - 'normal': Normal vector [a, b, c]
        """
        if len(points) == 0:
            return []
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 1. Downsample heavily (Speed is key here)
        # voxel_size: 5cm voxel size is enough for walls
        cloud = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        walls = []
        
        # "Iterative RANSAC"
        while True:
            if len(cloud.points) < self.min_points:
                break
            
            # 2. Try to find the biggest plane in the REMAINING cloud
            # distance_threshold: tolerance for points to be considered part of plane
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            # 3. STOPPING CONDITION: Is the plane too small?
            # If the biggest plane has fewer than min_points, it's just noise/furniture.
            if len(inliers) < self.min_points:
                break
            
            # 4. Check Orientation (Is it a Wall or Floor?)
            [a, b, c, d] = plane_model
            
            # A Wall is vertical. Its normal vector (a, b, c) should point sideways.
            # So 'c' (the up/down component) should be near 0.
            if abs(c) < self.vertical_threshold:  
                # It is a Wall!
                wall_pcd = cloud.select_by_index(inliers)
                
                # Store wall information
                wall_points = np.asarray(wall_pcd.points)
                wall_info = {
                    'plane_model': plane_model,
                    'points': wall_points,
                    'num_points': len(inliers),
                    'normal': np.array([a, b, c])
                }
                walls.append(wall_info)
            
            # 5. Remove the points we just found (Wall OR Floor)
            # Whether it was a wall or a floor, we remove it so we can find the NEXT plane.
            cloud = cloud.select_by_index(inliers, invert=True)
        
        return walls
