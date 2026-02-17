#!/usr/bin/env python3

import open3d as o3d
import numpy as np

class PersonDetector:
    """
    Detects standing/walking persons using dimensional constraints.
    """
    
    def __init__(self, 
                 min_human_height=1.40,
                 max_human_width=0.80,
                 min_points=50,
                 eps=0.25,
                 max_height=2.5):
        """
        Initialize person detector with tuning parameters.
        
        Args:
            min_human_height: Minimum height for human detection (meters)
            max_human_width: Maximum width for human detection (meters)
            min_points: Minimum points in cluster to consider
            eps: DBSCAN clustering tolerance (meters)
            max_height: Maximum height to consider (cap at ceiling)
        """
        self.min_human_height = min_human_height
        self.max_human_width = max_human_width
        self.min_points = min_points
        self.eps = eps
        self.max_height = max_height
        
    def detect(self, pcd):
        """
        Detect persons in point cloud.
        
        Args:
            pcd: Open3D PointCloud object
            
        Returns:
            List of dictionaries containing person detection info:
            [{
                'center': [x, y, z],
                'height': float,
                'width': float,
                'points': int,
                'cluster': PointCloud
            }, ...]
        """
        # 1. Remove floor (keep points above 10cm, below max_height)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-100, -100, 0.1]), 
            max_bound=np.array([100, 100, self.max_height])
        )
        no_floor = pcd.crop(bbox)
        
        if not no_floor.has_points():
            return []
        
        # 2. Cluster objects using DBSCAN
        labels = np.array(no_floor.cluster_dbscan(eps=self.eps, min_points=self.min_points))
        
        if len(labels) == 0:
            return []
            
        max_label = labels.max()
        detections = []
        
        # 3. Check each cluster against human criteria
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster = no_floor.select_by_index(cluster_indices)
            
            # Apply human filter
            result = self._is_human(cluster)
            if result is not None:
                detections.append(result)
                
        return detections
    
    def _is_human(self, cluster):
        """
        Check if cluster matches human dimensions.
        
        Args:
            cluster: Open3D PointCloud
            
        Returns:
            Dictionary with detection info if human, None otherwise
        """
        # Get bounding box
        aabb = cluster.get_axis_aligned_bounding_box()
        min_b = aabb.get_min_bound()
        max_b = aabb.get_max_bound()
        
        # Calculate dimensions
        height = max_b[2] - min_b[2]
        width_x = max_b[0] - min_b[0]
        width_y = max_b[1] - min_b[1]
        max_width = max(width_x, width_y)
        num_points = len(cluster.points)
        
        # Debug logging for clusters that don't pass
        import rospy
        
        # RULE 1: Height check (humans > 1.4m, robots/trays < 1.0m)
        if height < self.min_human_height:
            rospy.logdebug(f"Cluster rejected: height {height:.2f}m < {self.min_human_height}m (pts={num_points})")
            return None
            
        # RULE 2: Width check (walls are wide, humans are narrow)
        if max_width > self.max_human_width:
            rospy.logdebug(f"Cluster rejected: width {max_width:.2f}m > {self.max_human_width}m (h={height:.2f}m, pts={num_points})")
            return None
            
        # RULE 3: Aspect ratio check (humans are tall cylinders)
        # Height should be greater than width
        if height < max_width:
            rospy.logdebug(f"Cluster rejected: bad aspect ratio h={height:.2f}m < w={max_width:.2f}m (pts={num_points})")
            return None
        
        # Passed all checks - it's a human!
        rospy.logdebug(f"Person detected: h={height:.2f}m, w={max_width:.2f}m, pts={num_points}")
        center = cluster.get_center()
        
        return {
            'center': center.tolist(),
            'height': float(height),
            'width': float(max_width),
            'points': int(num_points),
            'cluster': cluster,
            'min_bound': min_b.tolist(),
            'max_bound': max_b.tolist()
        }
