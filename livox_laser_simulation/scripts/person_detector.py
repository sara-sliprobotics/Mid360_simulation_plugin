#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import time

class PersonDetector:
    """
    Detects standing/walking persons using dimensional constraints and motion detection.
    """
    
    def __init__(self, 
                 min_human_height=1.40,
                 max_human_width=0.80,
                 min_points=50,
                 eps=0.25,
                 max_height=2.5,
                 require_motion=True,
                 motion_threshold=0.1,
                 stationary_timeout=2.0):
        """
        Initialize person detector with tuning parameters.
        
        Args:
            min_human_height: Minimum height for human detection (meters)
            max_human_width: Maximum width for human detection (meters)
            min_points: Minimum points in cluster to consider
            eps: DBSCAN clustering tolerance (meters)
            max_height: Maximum height to consider (cap at ceiling)
            require_motion: If True, only detect moving persons
            motion_threshold: Minimum distance moved to be considered moving (meters)
            stationary_timeout: Time before a stationary person is filtered out (seconds)
        """
        self.min_human_height = min_human_height
        self.max_human_width = max_human_width
        self.min_points = min_points
        self.eps = eps
        self.max_height = max_height
        self.require_motion = require_motion
        self.motion_threshold = motion_threshold
        self.stationary_timeout = stationary_timeout
        
        # Track detections over time for motion detection
        # Format: {person_id: {'position': [x,y,z], 'last_update': timestamp, 'last_moved': timestamp}}
        self.tracked_persons = {}
        
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
                'cluster': PointCloud,
                'is_moving': bool
            }, ...]
        """
        current_time = time.time()
        
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
        raw_detections = []
        
        # 3. Check each cluster against human criteria
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster = no_floor.select_by_index(cluster_indices)
            
            # Apply human filter
            result = self._is_human(cluster)
            if result is not None:
                raw_detections.append(result)
        
        # 4. Apply motion filtering if enabled
        if self.require_motion:
            detections = self._filter_by_motion(raw_detections, current_time)
        else:
            detections = raw_detections
                
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
    def _filter_by_motion(self, detections, current_time):
        """
        Filter detections to only include moving persons.
        
        Args:
            detections: List of detection dictionaries
            current_time: Current timestamp
            
        Returns:
            Filtered list of detections (only moving persons)
        """
        import rospy
        
        # Clean up old tracks
        to_remove = []
        for person_id in list(self.tracked_persons.keys()):
            if current_time - self.tracked_persons[person_id]['last_update'] > 5.0:
                to_remove.append(person_id)
        for person_id in to_remove:
            del self.tracked_persons[person_id]
        
        moving_detections = []
        
        for detection in detections:
            center = np.array(detection['center'])
            
            # Try to match with existing tracked persons
            matched = False
            best_match_id = None
            best_match_dist = float('inf')
            
            for person_id, track in self.tracked_persons.items():
                tracked_pos = np.array(track['position'])
                dist = np.linalg.norm(center[:2] - tracked_pos[:2])  # XY distance only
                
                # If within 1 meter, consider it the same person
                if dist < 1.0 and dist < best_match_dist:
                    best_match_id = person_id
                    best_match_dist = dist
                    matched = True
            
            if matched:
                # Update existing track
                track = self.tracked_persons[best_match_id]
                old_pos = np.array(track['position'])
                movement = np.linalg.norm(center[:2] - old_pos[:2])
                
                track['last_update'] = current_time
                
                if movement > self.motion_threshold:
                    # Person has moved!
                    track['position'] = center.tolist()
                    track['last_moved'] = current_time
                    detection['is_moving'] = True
                    moving_detections.append(detection)
                    rospy.loginfo_throttle(1.0, f"Person {best_match_id} moved {movement:.2f}m - DETECTED")
                else:
                    # Person hasn't moved enough
                    time_stationary = current_time - track['last_moved']
                    if time_stationary < self.stationary_timeout:
                        # Still within grace period, include them
                        detection['is_moving'] = True
                        moving_detections.append(detection)
                        rospy.loginfo_throttle(1.0, f"Person {best_match_id} stationary for {time_stationary:.1f}s - still detected (grace period)")
                    else:
                        # Been stationary too long, filter out
                        detection['is_moving'] = False
                        rospy.loginfo_throttle(1.0, f"Person {best_match_id} stationary for {time_stationary:.1f}s - FILTERED OUT")
            else:
                # New detection - add to tracking
                new_id = len(self.tracked_persons)
                self.tracked_persons[new_id] = {
                    'position': center.tolist(),
                    'last_update': current_time,
                    'last_moved': current_time  # Assume new detections are moving
                }
                detection['is_moving'] = True
                moving_detections.append(detection)
                rospy.loginfo(f"New person {new_id} detected at {center[:2]} - DETECTED (new)")
        
        return moving_detections