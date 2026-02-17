#!/usr/bin/env python3

import numpy as np


class TruckDetector:
    """Detects trucks by finding two parallel vertical walls with specific spacing"""
    
    def __init__(self, truck_width=2.6, width_tolerance=0.3, parallel_angle_threshold=0.1, 
                 min_wall_length=3.0):
        """
        Initialize truck detector with parameters
        
        Args:
            truck_width: Expected width between truck walls (m)
            width_tolerance: Tolerance for width matching (m)
            parallel_angle_threshold: Max angle difference for parallel walls (radians)
            min_wall_length: Minimum length for walls to be considered truck walls (m)
        """
        self.truck_width = truck_width
        self.width_tolerance = width_tolerance
        self.parallel_angle_threshold = parallel_angle_threshold
        self.min_wall_length = min_wall_length
    
    def detect(self, walls):
        """
        Detect trucks from detected wall pairs
        
        Args:
            walls: List of wall dictionaries from WallDetector with keys:
                - 'plane_model': [a, b, c, d] plane equation
                - 'points': Nx3 numpy array of points on the wall
                - 'normal': Normal vector [a, b, c]
                
        Returns:
            List of truck dictionaries with keys:
                - 'wall_pair': Indices of the two walls forming the truck
                - 'center': Center position between the walls
                - 'width': Actual measured width
                - 'orientation': Average normal direction
        """
        if len(walls) < 2:
            return []
        
        trucks = []
        used_walls = set()
        
        # Check all pairs of walls
        for i in range(len(walls)):
            if i in used_walls:
                continue
                
            for j in range(i + 1, len(walls)):
                if j in used_walls:
                    continue
                
                wall_i = walls[i]
                wall_j = walls[j]
                
                # Check if walls are parallel
                if not self._are_parallel(wall_i['normal'], wall_j['normal']):
                    continue
                
                # Check if walls are long enough (truck walls should be long)
                if not self._are_long_enough(wall_i, wall_j):
                    continue
                
                # Calculate distance between parallel walls
                distance = self._calculate_wall_distance(wall_i, wall_j)
                
                # Check if distance matches truck width
                if abs(distance - self.truck_width) <= self.width_tolerance:
                    # Found a truck!
                    # Calculate centers of both walls
                    center_i = np.mean(wall_i['points'], axis=0)
                    center_j = np.mean(wall_j['points'], axis=0)
                    
                    # Determine which wall is closer to sensor (front wall)
                    sensor_pos = np.array([0.0, 0.0, 0.0])
                    dist_i = np.linalg.norm(center_i[:2] - sensor_pos[:2])
                    dist_j = np.linalg.norm(center_j[:2] - sensor_pos[:2])
                    
                    if dist_i < dist_j:
                        front_wall = wall_i
                        back_wall = wall_j
                        front_center = center_i
                        back_center = center_j
                    else:
                        front_wall = wall_j
                        back_wall = wall_i
                        front_center = center_j
                        back_center = center_i
                    
                    # Calculate truck length from wall dimensions
                    front_points = front_wall['points']
                    
                    # Use truck model dimensions (from model.sdf)
                    truck_length = 16.14  # Truck length from model (meters)
                    
                    # Calculate wall direction (perpendicular to normal, along the wall)
                    wall_normal = front_wall['normal']
                    wall_direction = np.array([-wall_normal[1], wall_normal[0], 0.0])
                    if np.linalg.norm(wall_direction) > 0:
                        wall_direction = wall_direction / np.linalg.norm(wall_direction)
                    
                    # Find the front EDGE - closest point on front wall to sensor
                    distances_to_sensor = np.linalg.norm(front_points[:, :2] - sensor_pos[:2], axis=1)
                    closest_idx = np.argmin(distances_to_sensor)
                    front_edge_point = front_points[closest_idx]
                    
                    # Calculate wall direction (perpendicular to normal, along the wall)
                    wall_normal = front_wall['normal']
                    wall_direction = np.array([-wall_normal[1], wall_normal[0], 0.0])
                    if np.linalg.norm(wall_direction) > 0:
                        wall_direction = wall_direction / np.linalg.norm(wall_direction)
                    
                    # Direction along the wall from front edge toward wall center
                    to_wall_center = front_center - front_edge_point
                    to_wall_center[2] = 0
                    if np.linalg.norm(to_wall_center) > 0:
                        length_direction = to_wall_center / np.linalg.norm(to_wall_center)
                    else:
                        length_direction = wall_direction
                    
                    # Direction from front wall to back wall (perpendicular to wall)
                    front_to_back = back_center - front_center
                    front_to_back[2] = 0  # Keep in XY plane
                    if np.linalg.norm(front_to_back) > 0:
                        depth_direction = front_to_back / np.linalg.norm(front_to_back)
                    else:
                        depth_direction = np.array([0.0, 0.0, 0.0])
                    
                    # Truck center: 
                    # 1. Start from front edge
                    # 2. Move along wall by half truck length
                    # 3. Move perpendicular to wall by half truck width
                    truck_center_xy = front_edge_point[:2] + length_direction[:2] * (truck_length / 2.0) + depth_direction[:2] * (distance / 2.0)
                    truck_center = np.array([truck_center_xy[0], truck_center_xy[1], 1.25])
                    
                    truck_info = {
                        'wall_pair': (i, j),
                        'center': truck_center,
                        'front_edge': front_edge_point,
                        'width': distance,
                        'length': truck_length,
                        'height': 2.5,
                        'orientation': wall_normal,
                        'direction': wall_direction,
                        'walls': [wall_i, wall_j]
                    }
                    trucks.append(truck_info)
                    
                    # Mark walls as used
                    used_walls.add(i)
                    used_walls.add(j)
                    break
        
        return trucks
    
    def _are_parallel(self, normal1, normal2):
        """
        Check if two wall normals are parallel (or anti-parallel)
        
        Args:
            normal1, normal2: Normal vectors [a, b, c]
            
        Returns:
            True if walls are parallel
        """
        # Normalize
        n1 = normal1 / np.linalg.norm(normal1)
        n2 = normal2 / np.linalg.norm(normal2)
        
        # Dot product should be close to 1 (parallel) or -1 (anti-parallel)
        dot = abs(np.dot(n1, n2))
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        return angle < self.parallel_angle_threshold
    
    def _are_long_enough(self, wall_i, wall_j):
        """
        Check if both walls are long enough to be truck walls
        
        Args:
            wall_i, wall_j: Wall dictionaries
            
        Returns:
            True if both walls meet minimum length
        """
        for wall in [wall_i, wall_j]:
            points = wall['points']
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            
            # Check length in X and Y directions
            length_x = max_pt[0] - min_pt[0]
            length_y = max_pt[1] - min_pt[1]
            max_length = max(length_x, length_y)
            
            if max_length < self.min_wall_length:
                return False
        
        return True
    
    def _calculate_wall_distance(self, wall_i, wall_j):
        """
        Calculate perpendicular distance between two parallel walls
        
        Args:
            wall_i, wall_j: Wall dictionaries with 'plane_model' and 'points'
            
        Returns:
            Distance between walls (m)
        """
        # Use plane equations: ax + by + cz + d = 0
        # Distance between parallel planes = |d1 - d2| / sqrt(a^2 + b^2 + c^2)
        
        a1, b1, c1, d1 = wall_i['plane_model']
        a2, b2, c2, d2 = wall_j['plane_model']
        
        # Normalize coefficients
        norm1 = np.sqrt(a1**2 + b1**2 + c1**2)
        norm2 = np.sqrt(a2**2 + b2**2 + c2**2)
        
        d1_norm = d1 / norm1
        d2_norm = d2 / norm2
        
        # Calculate distance
        distance = abs(d1_norm - d2_norm)
        
        return distance
