#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
import json
import sys
import os

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from leg_detector import LegDetector
from wall_detector import WallDetector


class LivoxObjectDetector:
    def __init__(self):
        rospy.init_node('livox_object_detector', anonymous=True)
        
        # Parameters for leg detection
        self.min_points = rospy.get_param('~min_points', 50)
        leg_height_min = rospy.get_param('~leg_height_min', 0.1)
        leg_height_max = rospy.get_param('~leg_height_max', 0.5)
        leg_width_min = rospy.get_param('~leg_width_min', 0.08)
        leg_width_max = rospy.get_param('~leg_width_max', 0.25)
        dbscan_eps = rospy.get_param('~dbscan_eps', 0.10)
        dbscan_min_samples = rospy.get_param('~dbscan_min_samples', 10)
        
        # Parameters for wall detection
        wall_voxel_size = rospy.get_param('~wall_voxel_size', 0.05)
        wall_distance_threshold = rospy.get_param('~wall_distance_threshold', 0.1)
        wall_min_points = rospy.get_param('~wall_min_points', 200)
        wall_vertical_threshold = rospy.get_param('~wall_vertical_threshold', 0.2)
        wall_min_width = rospy.get_param('~wall_min_width', 2.0)
        
        # Initialize detectors
        self.leg_detector = LegDetector(
            height_min=leg_height_min,
            height_max=leg_height_max,
            width_min=leg_width_min,
            width_max=leg_width_max,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples
        )
        
        self.wall_detector = WallDetector(
            voxel_size=wall_voxel_size,
            distance_threshold=wall_distance_threshold,
            min_points=wall_min_points,
            vertical_threshold=wall_vertical_threshold,
            min_wall_width=wall_min_width
        )
        
        # Subscriber
        self.pc_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.pointcloud_callback)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/livox/detected_objects', MarkerArray, queue_size=10)
        self.detection_pub = rospy.Publisher('/livox/detections', String, queue_size=10)
        
        rospy.loginfo("Livox Object Detector initialized with Leg and Wall Detection")
        rospy.loginfo(f"Leg height: {leg_height_min}-{leg_height_max}m, width: {leg_width_min}-{leg_width_max}m")
        rospy.loginfo(f"Wall detection: voxel={wall_voxel_size}m, min_points={wall_min_points}")
    
    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            if len(points) < self.min_points:
                rospy.logwarn_throttle(5.0, f"Too few points: {len(points)}")
                return
            
            # Detect legs using leg detector
            human_locations = self.leg_detector.detect(points)
            
            # Detect walls using wall detector
            walls = self.wall_detector.detect(points)
            
            # Publish detection results as structured data
            self.publish_detections(human_locations, walls, msg.header)
            
            # Visualize detected humans and walls
            self.visualize_detections(human_locations, walls, msg.header)
            
            rospy.loginfo_throttle(1.0, f"Detected {len(human_locations)} human(s) and {len(walls)} wall(s)")
            
        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        points_list = []
        
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            points_list.append([point[0], point[1], point[2]])
        
        return np.array(points_list)
    
    def publish_detections(self, human_locations, walls, header):
        """Publish structured detection results as JSON"""
        detections = {
            'timestamp': header.stamp.to_sec(),
            'frame_id': header.frame_id,
            'humans': [],
            'walls': []
        }
        
        # Add human detections
        for i, center in enumerate(human_locations):
            detections['humans'].append({
                'id': i,
                'type': 'human',
                'position': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'z': float(center[2])
                }
            })
        
        # Add wall detections
        for i, wall in enumerate(walls):
            wall_points = wall['points']
            min_pt = np.min(wall_points, axis=0)
            max_pt = np.max(wall_points, axis=0)
            center = (min_pt + max_pt) / 2.0
            size = max_pt - min_pt
            
            detections['walls'].append({
                'id': i,
                'type': 'wall',
                'center': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'z': float(center[2])
                },
                'size': {
                    'x': float(size[0]),
                    'y': float(size[1]),
                    'z': float(size[2])
                },
                'normal': {
                    'x': float(wall['normal'][0]),
                    'y': float(wall['normal'][1]),
                    'z': float(wall['normal'][2])
                },
                'num_points': wall['num_points']
            })
        
        # Publish as JSON string
        msg = String()
        msg.data = json.dumps(detections, indent=2)
        self.detection_pub.publish(msg)
    
    def visualize_detections(self, human_locations, walls, header):
        """Publish visualization markers for detected humans and walls"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # Visualize humans
        for i, center in enumerate(human_locations):
            # Cylinder marker for human (approximate height)
            marker = Marker()
            marker.header = header
            marker.ns = "detected_humans"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position (center of detected leg at ground level, extend upward)
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = 0.85  # Approximate torso center
            marker.pose.orientation.w = 1.0
            
            # Size (approximate human dimensions)
            marker.scale.x = 0.4  # Width
            marker.scale.y = 0.4  # Depth
            marker.scale.z = 1.7  # Height
            
            # Color (green for human)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(marker)
            
            # Text label at detected leg position
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "human_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = center[0]
            text_marker.pose.position.y = center[1]
            text_marker.pose.position.z = 1.8  # Above head
            
            text_marker.text = f"Human\n({center[0]:.1f}, {center[1]:.1f})"
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(text_marker)
            
            # Sphere marker at detected leg position (exact location)
            sphere_marker = Marker()
            sphere_marker.header = header
            sphere_marker.ns = "leg_positions"
            sphere_marker.id = marker_id
            marker_id += 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            sphere_marker.pose.position.x = center[0]
            sphere_marker.pose.position.y = center[1]
            sphere_marker.pose.position.z = center[2]
            sphere_marker.pose.orientation.w = 1.0
            
            sphere_marker.scale.x = 0.15
            sphere_marker.scale.y = 0.15
            sphere_marker.scale.z = 0.15
            
            sphere_marker.color.r = 1.0
            sphere_marker.color.g = 0.0
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 0.8
            sphere_marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(sphere_marker)
        
        # Visualize walls
        for i, wall in enumerate(walls):
            wall_points = wall['points']
            
            # Calculate wall bounds
            min_pt = np.min(wall_points, axis=0)
            max_pt = np.max(wall_points, axis=0)
            center = (min_pt + max_pt) / 2.0
            size = max_pt - min_pt
            
            # Wall bounding box marker
            wall_marker = Marker()
            wall_marker.header = header
            wall_marker.ns = "detected_walls"
            wall_marker.id = marker_id
            marker_id += 1
            wall_marker.type = Marker.CUBE
            wall_marker.action = Marker.ADD
            
            # Position
            wall_marker.pose.position.x = center[0]
            wall_marker.pose.position.y = center[1]
            wall_marker.pose.position.z = center[2]
            wall_marker.pose.orientation.w = 1.0
            
            # Size
            wall_marker.scale.x = max(size[0], 0.05)
            wall_marker.scale.y = max(size[1], 0.05)
            wall_marker.scale.z = max(size[2], 0.05)
            
            # Color (red for walls)
            wall_marker.color.r = 1.0
            wall_marker.color.g = 0.0
            wall_marker.color.b = 0.0
            wall_marker.color.a = 0.3
            wall_marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(wall_marker)
            
            # Wall text label
            wall_text = Marker()
            wall_text.header = header
            wall_text.ns = "wall_labels"
            wall_text.id = marker_id
            marker_id += 1
            wall_text.type = Marker.TEXT_VIEW_FACING
            wall_text.action = Marker.ADD
            
            wall_text.pose.position.x = center[0]
            wall_text.pose.position.y = center[1]
            wall_text.pose.position.z = max_pt[2] + 0.3
            
            normal = wall['normal']
            wall_text.text = f"Wall #{i+1}\n{wall['num_points']} pts\nN:({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"
            wall_text.scale.z = 0.2
            wall_text.color.r = 1.0
            wall_text.color.g = 1.0
            wall_text.color.b = 1.0
            wall_text.color.a = 1.0
            wall_text.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(wall_text)
        
        self.marker_pub.publish(marker_array)
    
    def run(self):
        """Main loop"""
        rospy.spin()


if __name__ == '__main__':
    try:
        detector = LivoxObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
