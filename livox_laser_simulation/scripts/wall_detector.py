#!/usr/bin/env python3

import numpy as np
import open3d as o3d


class WallDetector:
    """Detects walls from point cloud data using RANSAC plane segmentation"""
    
    def __init__(self, voxel_size=0.05, distance_threshold=0.1, min_points=200, 
                 vertical_threshold=0.2, ransac_n=3, num_iterations=1000, min_wall_width=2.0,
                 connectivity_eps=0.5):
        """
        Initialize wall detector with parameters
        
        Args:
            voxel_size: Voxel size for downsampling (m)
            distance_threshold: RANSAC distance threshold (m)
            min_points: Minimum points for a valid plane
            vertical_threshold: Maximum vertical component of normal for walls
            ransac_n: Number of points for RANSAC
            num_iterations: RANSAC iterations
            min_wall_width: Minimum width for a plane to be considered a wall (m)
            connectivity_eps: Max distance for spatial clustering to separate disconnected objects (m)
        """
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.min_points = min_points
        self.vertical_threshold = vertical_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.min_wall_width = min_wall_width
        self.connectivity_eps = connectivity_eps
    
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
                
                # CRITICAL: Check spatial connectivity to avoid merging wall + separated objects
                # Even if they fit the same plane equation, they should be separate if not connected
                main_cluster = self._get_largest_connected_cluster(wall_pcd)
                
                if len(main_cluster.points) < self.min_points:
                    # After filtering, not enough points - skip this plane
                    cloud = cloud.select_by_index(inliers, invert=True)
                    continue
                
                wall_points = np.asarray(main_cluster.points)
                
                # Check wall dimensions - reject if too narrow
                min_pt = np.min(wall_points, axis=0)
                max_pt = np.max(wall_points, axis=0)
                width_x = max_pt[0] - min_pt[0]
                width_y = max_pt[1] - min_pt[1]
                max_width = max(width_x, width_y)
                
                # Only keep if wide enough to be a real wall
                if max_width >= self.min_wall_width:
                    # Store wall information
                    wall_info = {
                        'plane_model': plane_model,
                        'points': wall_points,
                        'num_points': len(wall_points),
                        'normal': np.array([a, b, c])
                    }
                    walls.append(wall_info)
            
            # 5. Remove the points we just found (Wall OR Floor)
            # Whether it was a wall or a floor, we remove it so we can find the NEXT plane.
            cloud = cloud.select_by_index(inliers, invert=True)
        
        return walls
    
    def _get_largest_connected_cluster(self, pcd):
        """
        Use DBSCAN to find spatially connected clusters and return the largest one.
        This separates objects that fit the same plane but are disconnected (wall + tray).
        
        Args:
            pcd: Open3D PointCloud
            
        Returns:
            Open3D PointCloud containing only the largest connected cluster
        """
        # Cluster points spatially using DBSCAN
        labels = np.array(pcd.cluster_dbscan(eps=self.connectivity_eps, min_points=5))
        
        # If no valid clusters or all noise, return original
        if labels.size == 0 or np.all(labels < 0):
            return pcd
        
        # Find largest cluster (ignoring noise points with label -1)
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        if len(valid_labels) == 0:
            return pcd
        
        # Count points in each cluster and get the largest
        largest_cluster_label = None
        largest_cluster_size = 0
        
        for label in valid_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_label = label
        
        # Extract only the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        return pcd.select_by_index(largest_cluster_indices)
