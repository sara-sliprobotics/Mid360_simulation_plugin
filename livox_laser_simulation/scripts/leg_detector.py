#!/usr/bin/env python3

import numpy as np
import open3d as o3d


class LegDetector:
    """Detects human legs from point cloud data"""
    
    def __init__(self, height_min=0.1, height_max=0.5, width_min=0.08, width_max=0.25, 
                 eps=0.10, min_samples=10):
        """
        Initialize leg detector with parameters
        
        Args:
            height_min: Minimum height to slice (m)
            height_max: Maximum height to slice (m)
            width_min: Minimum leg width (m)
            width_max: Maximum leg width (m)
            eps: DBSCAN clustering distance (m)
            min_samples: DBSCAN minimum points per cluster
        """
        self.height_min = height_min
        self.height_max = height_max
        self.width_min = width_min
        self.width_max = width_max
        self.eps = eps
        self.min_samples = min_samples
    
    def detect(self, points):
        """
        Detect human legs using Open3D clustering
        
        Args:
            points: Nx3 numpy array of 3D points
            
        Returns:
            List of detected leg center positions (3D points)
        """
        if len(points) == 0:
            return []
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 1. Height Slicing (Region of Interest)
        # Keep only points between height_min and height_max (Shin/Knee level)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-10, -10, self.height_min]), 
            max_bound=np.array([ 10,  10, self.height_max])
        )
        cropped_pcd = pcd.crop(bbox)
        
        if len(cropped_pcd.points) == 0:
            return []
        
        # 2. Clustering (DBSCAN)
        # eps means points within that distance are part of the same object
        # min_samples removes tiny noise specs
        labels = np.array(cropped_pcd.cluster_dbscan(
            eps=self.eps, 
            min_points=self.min_samples, 
            print_progress=False
        ))
        
        if len(labels) == 0:
            return []
        
        max_label = labels.max()
        human_locations = []
        
        # 3. Analyze each cluster
        for i in range(max_label + 1):
            # Extract points for this specific cluster
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = cropped_pcd.select_by_index(cluster_indices)
            
            # Get bounds (Size of the object)
            min_b = cluster_pcd.get_min_bound()
            max_b = cluster_pcd.get_max_bound()
            
            width_x = max_b[0] - min_b[0]
            width_y = max_b[1] - min_b[1]
            
            # 4. The "Leg Test"
            # Legs are usually roughly width_min to width_max thick
            width = max(width_x, width_y)
            if self.width_min < width < self.width_max:
                # It's a leg! Save the center position.
                center = cluster_pcd.get_center()
                human_locations.append(center)
        
        return human_locations
