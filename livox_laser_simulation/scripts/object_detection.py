#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String, Bool
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

from wall_detector import WallDetector
from truck_detector import TruckDetector
from person_detector import PersonDetector
from tray_detector import LegFirstTrayDetector
from leg_detector import LegDetector
from leg_model import create_leg_markers
from frame_accumulator import FrameAccumulator


class LivoxObjectDetector:
    def __init__(self):
        rospy.init_node('livox_object_detector', anonymous=True)
        
        # Parameters
        self.min_points = rospy.get_param('~min_points', 50)
        
        # Parameters for wall detection
        wall_voxel_size = rospy.get_param('~wall_voxel_size', 0.05)
        wall_distance_threshold = rospy.get_param('~wall_distance_threshold', 0.1)
        wall_min_points = rospy.get_param('~wall_min_points', 200)
        wall_vertical_threshold = rospy.get_param('~wall_vertical_threshold', 0.2)
        wall_min_width = rospy.get_param('~wall_min_width', 2.0)
        
        # Parameters for truck detection
        truck_width = rospy.get_param('~truck_width', 2.6)
        truck_width_tolerance = rospy.get_param('~truck_width_tolerance', 0.3)
        truck_min_wall_length = rospy.get_param('~truck_min_wall_length', 3.0)
        
        # Parameters for person detection
        person_min_height = rospy.get_param('~person_min_height', 1.30)  # Lowered from 1.40
        person_max_width = rospy.get_param('~person_max_width', 0.80)
        person_min_points = rospy.get_param('~person_min_points', 30)  # Lowered from 50
        person_eps = rospy.get_param('~person_eps', 0.35)  # Increased from 0.25
        person_require_motion = rospy.get_param('~person_require_motion', True)
        person_motion_threshold = rospy.get_param('~person_motion_threshold', 0.1)
        person_stationary_timeout = rospy.get_param('~person_stationary_timeout', 2.0)

        # Initialize detectors
        self.wall_detector = WallDetector(
            voxel_size=wall_voxel_size,
            distance_threshold=wall_distance_threshold,
            min_points=wall_min_points,
            vertical_threshold=wall_vertical_threshold,
            min_wall_width=wall_min_width
        )
        
        self.truck_detector = TruckDetector(
            truck_width=truck_width,
            width_tolerance=truck_width_tolerance,
            min_wall_length=truck_min_wall_length
        )
        
        self.person_detector = PersonDetector(
            min_human_height=person_min_height,
            max_human_width=person_max_width,
            min_points=person_min_points,
            eps=person_eps,
            require_motion=person_require_motion,
            motion_threshold=person_motion_threshold,
            stationary_timeout=person_stationary_timeout
        )
        
        # Initialize FrameAccumulator (accumulate frames with TF into fixed frame)
        accumulator_frames = rospy.get_param('~accumulator_frames', 3)
        self.accumulator = FrameAccumulator('/livox/lidar', num_frames=accumulator_frames, fixed_frame='odom')

        # Initialize tray and leg detectors with the shared accumulator
        self.tray_detector = LegFirstTrayDetector(accumulator=self.accumulator)
        self.leg_detector = LegDetector(accumulator=self.accumulator)

        # Entering-tray mode: when True, also run leg detection and persist tray box
        self._entering_tray = False
        self._tracked_trays = []
        rospy.Subscriber('/entering_tray', Bool, self._entering_tray_cb)

        # Subscriber - use queue_size=1 to only process latest, drop old messages
        # Large buff_size to handle large point cloud messages
        self.pc_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.pointcloud_callback, 
                                       queue_size=1, buff_size=2**24)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/livox/detected_objects', MarkerArray, queue_size=10)
        self.detection_pub = rospy.Publisher('/livox/detections', String, queue_size=10)
        self.accumulated_cloud_pub = rospy.Publisher('/livox/accumulated_cloud', PointCloud2, queue_size=1)
        
        rospy.loginfo("Livox Object Detector initialized with Wall, Truck, Person, and Tray Detection")
        rospy.loginfo(f"Wall detection: voxel={wall_voxel_size}m, min_points={wall_min_points}")
        rospy.loginfo(f"Truck detection: width={truck_width}m ± {truck_width_tolerance}m")
        rospy.loginfo(f"Person detection: height>={person_min_height}m, width<={person_max_width}m, motion_required={person_require_motion}")
        if person_require_motion:
            rospy.loginfo(f"  Motion filter: threshold={person_motion_threshold}m, timeout={person_stationary_timeout}s")
        rospy.loginfo("Tray detection: leg-based detection enabled")
        rospy.loginfo(f"Frame accumulator: {accumulator_frames} frames")
    
    def _entering_tray_cb(self, msg):
        if msg.data and not self._entering_tray:
            rospy.loginfo("[TRAY TRACKING] Entering tray mode activated")
        elif not msg.data and self._entering_tray:
            rospy.loginfo("[TRAY TRACKING] Entering tray mode deactivated")
            self._tracked_trays = []
        self._entering_tray = msg.data

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        try:
            import time
            callback_start = time.time()

            walls = []
            trucks = []
            persons = []
            trays = []
            legs = []

            if not self._entering_tray:
                # Normal mode: detect walls, trucks, persons, trays (no legs)
                points = self.pointcloud2_to_array(msg)
                if len(points) < self.min_points:
                    rospy.logwarn_throttle(5.0, f"Too few points: {len(points)}")
                    return

                walls = self.wall_detector.detect(points)
                trucks = self.truck_detector.detect(walls)
                persons = self.person_detector.detect(self.numpy_to_open3d(points))
                trays = self.tray_detector.detect(walls=walls)
                if trays:
                    self._tracked_trays = trays
            else:
                # # Entering tray mode: only detect legs, skip everything else
                # # TEMP TEST: inject fake tray so we skip detection and go straight to legs
                # if not self._tracked_trays:
                #     self._tracked_trays = [{
                #         'type': 'FAKE_TEST',
                #         'center': np.array([0.0, -4.0, 0.38]),
                #         'legs': [],
                #         'orientation': np.array([0.0, -1.0, 0.0]),
                #     }]
                #     rospy.loginfo("[TRAY TRACKING] TEMP: Using fake tray for testing")

                # If no saved tray yet, detect tray once then lock it in
                if not self._tracked_trays:
                    points = self.pointcloud2_to_array(msg)
                    if len(points) < self.min_points:
                        return
                    walls = self.wall_detector.detect(points)
                    detected = self.tray_detector.detect(walls=walls)
                    if detected:
                        self._tracked_trays = [detected[0]]  # Save only the best tray
                        rospy.loginfo(f"[TRAY TRACKING] Tray locked: {detected[0].get('type', 'UNKNOWN')}")
                    else:
                        rospy.loginfo_throttle(2.0, "[TRAY TRACKING] Searching for tray...")
                    return  # Don't visualize until tray is locked

                trays = self._tracked_trays

                for tray in trays:
                    tray_center = tray['center']

                    import tray_config
                    import open3d as o3d

                    inflation = 0.2
                    half_size = max(tray_config.TRAY_FULL_LENGTH, tray_config.TRAY_FULL_WIDTH) / 2.0 + inflation

                    min_bound = np.array([
                        tray_center[0] - half_size,
                        tray_center[1] - half_size,
                        0.0
                    ])
                    max_bound = np.array([
                        tray_center[0] + half_size,
                        tray_center[1] + half_size,
                        0.5
                    ])

                    tray_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    tray_legs = self.leg_detector.detect(box=tray_box)
                    legs.extend(tray_legs)

            # Publish accumulated cloud for debugging in RViz
            acc_cloud = self.accumulator.get_accumulated_cloud()
            if acc_cloud is not None and len(acc_cloud.points) > 0:
                acc_pts = np.asarray(acc_cloud.points).astype(np.float32)
                acc_header = Header()
                acc_header.stamp = msg.header.stamp
                acc_header.frame_id = 'odom'
                acc_msg = pc2.create_cloud_xyz32(acc_header, acc_pts)
                self.accumulated_cloud_pub.publish(acc_msg)

            total_time = time.time() - callback_start

            # Publish and visualize
            self.publish_detections(walls, trucks, persons, trays, legs, msg.header)
            self.visualize_detections(walls, trucks, persons, trays, legs, msg.header)

            rospy.loginfo_throttle(1.0, f"Processing: {total_time:.3f}s | trays={len(trays)} legs={len(legs)} walls={len(walls)} entering_tray={self._entering_tray}")
            
        except Exception as e:
            import traceback
            rospy.logerr(f"Error processing point cloud: {e}")
            rospy.logerr(traceback.format_exc())
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        points_list = []
        
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            points_list.append([point[0], point[1], point[2]])
        
        return np.array(points_list)
    
    def numpy_to_open3d(self, points):
        """Convert numpy array to Open3D PointCloud"""
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def publish_detections(self, walls, trucks, persons, trays, legs, header):
        """Publish structured detection results as JSON"""
        now = rospy.Time.now()
        detections = {
            'timestamp': header.stamp.to_sec(),
            'published_at': now.to_sec(),
            'frame_id': header.frame_id,
            'walls': [],
            'trucks': [],
            'persons': [],
            'trays': [],
            'legs': []
        }
        
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
        
        # Add truck detections
        for i, truck in enumerate(trucks):
            detections['trucks'].append({
                'id': i,
                'type': 'truck',
                'center': {
                    'x': float(truck['center'][0]),
                    'y': float(truck['center'][1]),
                    'z': float(truck['center'][2])
                },
                'width': float(truck['width']),
                'orientation': {
                    'x': float(truck['orientation'][0]),
                    'y': float(truck['orientation'][1]),
                    'z': float(truck['orientation'][2])
                },
                'wall_indices': truck['wall_pair']
            })
        
        # Add person detections
        for i, person in enumerate(persons):
            detections['persons'].append({
                'id': i,
                'type': 'person',
                'center': {
                    'x': float(person['center'][0]),
                    'y': float(person['center'][1]),
                    'z': float(person['center'][2])
                },
                'height': person['height'],
                'width': person['width'],
                'num_points': person['points']
            })
        
        # Add tray detections
        for i, tray in enumerate(trays):
            detections['trays'].append({
                'id': i,
                'type': tray.get('type', 'TRAY'),
                'center': {
                    'x': float(tray['center'][0]),
                    'y': float(tray['center'][1]),
                    'z': float(tray['center'][2])
                },
                'num_legs': len(tray.get('legs', [])),
                'detection_type': tray.get('type', 'UNKNOWN')
            })
        
        # Add leg detections
        for i, leg in enumerate(legs):
            detections['legs'].append({
                'id': i,
                'type': 'leg_pair',
                'pair_type': leg['pair_type'],
                'center': {
                    'x': float(leg['center'][0]),
                    'y': float(leg['center'][1]),
                    'z': 0.0
                },
                'pose': {
                    'x': float(leg['pose']['x']),
                    'y': float(leg['pose']['y']),
                    'yaw': float(leg['pose']['yaw'])
                }
            })

        # Publish as JSON string
        msg = String()
        msg.data = json.dumps(detections, indent=2)
        self.detection_pub.publish(msg)
    
    def visualize_detections(self, walls, trucks, persons, trays, legs, header):
        """Publish visualization markers for detected walls, trucks, persons, trays, and legs"""
        marker_array = MarkerArray()
        marker_id = 0
        MARKER_LIFETIME = rospy.Duration(0.5)
        
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
            wall_marker.lifetime = MARKER_LIFETIME
            
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
            wall_text.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(wall_text)
        
        # Visualize trucks
        for i, truck in enumerate(trucks):
            center = truck['center']
            width = truck['width']
            length = truck['length']
            direction = truck['direction']
            
            # Calculate yaw angle from wall direction
            yaw = np.arctan2(direction[1], direction[0])
            
            # Convert to quaternion
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            
            # Bounding box for truck
            truck_marker = Marker()
            truck_marker.header = header
            truck_marker.ns = "trucks"
            truck_marker.id = marker_id
            marker_id += 1
            truck_marker.type = Marker.CUBE
            truck_marker.action = Marker.ADD
            
            truck_marker.pose.position.x = center[0]
            truck_marker.pose.position.y = center[1]
            truck_marker.pose.position.z = center[2]
            
            # Set orientation from wall direction
            truck_marker.pose.orientation.x = 0.0
            truck_marker.pose.orientation.y = 0.0
            truck_marker.pose.orientation.z = qz
            truck_marker.pose.orientation.w = qw
            
            # Truck dimensions: length along wall, width between walls
            truck_marker.scale.x = length  # Along wall direction
            truck_marker.scale.y = width   # Between parallel walls
            truck_marker.scale.z = 2.5     # Height
            
            # Color (blue for trucks)
            truck_marker.color.r = 0.0
            truck_marker.color.g = 0.5
            truck_marker.color.b = 1.0
            truck_marker.color.a = 0.4
            truck_marker.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(truck_marker)
            
            # Truck text label
            truck_text = Marker()
            truck_text.header = header
            truck_text.ns = "truck_labels"
            truck_text.id = marker_id
            marker_id += 1
            truck_text.type = Marker.TEXT_VIEW_FACING
            truck_text.action = Marker.ADD
            
            truck_text.pose.position.x = center[0]
            truck_text.pose.position.y = center[1]
            truck_text.pose.position.z = center[2] + 1.5
            
            truck_text.text = f"TRUCK #{i+1}\nW:{width:.2f}m L:{length:.2f}m"
            truck_text.scale.z = 0.3
            truck_text.color.r = 1.0
            truck_text.color.g = 1.0
            truck_text.color.b = 0.0
            truck_text.color.a = 1.0
            truck_text.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(truck_text)
        
        # Visualize persons
        for i, person in enumerate(persons):
            center = person['center']
            height = person['height']
            width = person['width']
            
            # Cylinder marker for person body
            person_marker = Marker()
            person_marker.header = header
            person_marker.ns = "detected_persons"
            person_marker.id = marker_id
            marker_id += 1
            person_marker.type = Marker.CYLINDER
            person_marker.action = Marker.ADD
            
            person_marker.pose.position.x = center[0]
            person_marker.pose.position.y = center[1]
            person_marker.pose.position.z = center[2]
            person_marker.pose.orientation.w = 1.0
            
            person_marker.scale.x = width
            person_marker.scale.y = width
            person_marker.scale.z = height
            person_marker.color.r = 1.0
            person_marker.color.g = 0.0
            person_marker.color.b = 0.0
            person_marker.color.a = 0.7
            person_marker.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(person_marker)
            
            # Text label for person
            person_text = Marker()
            person_text.header = header
            person_text.ns = "detected_persons_text"
            person_text.id = marker_id
            marker_id += 1
            person_text.type = Marker.TEXT_VIEW_FACING
            person_text.action = Marker.ADD
            
            person_text.pose.position.x = center[0]
            person_text.pose.position.y = center[1]
            person_text.pose.position.z = center[2] + height/2 + 0.3
            
            person_text.text = f"PERSON #{i+1}\nH:{height:.2f}m W:{width:.2f}m\n{person['points']} pts"
            person_text.scale.z = 0.25
            person_text.color.r = 1.0
            person_text.color.g = 1.0
            person_text.color.b = 1.0
            person_text.color.a = 1.0
            person_text.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(person_text)
        
        # Tray and leg detections are in odom frame (accumulated cloud),
        # so use an odom header instead of the base_link header
        odom_header = Header()
        odom_header.stamp = header.stamp
        odom_header.frame_id = 'odom'

        # Visualize trays
        for i, tray in enumerate(trays):
            center = tray['center']
            tray_type = tray.get('type', 'UNKNOWN')
            num_legs = len(tray.get('legs', []))

            # Tray platform marker (cube for the tray deck)
            tray_marker = Marker()
            tray_marker.header = odom_header
            tray_marker.ns = "detected_trays"
            tray_marker.id = marker_id
            marker_id += 1
            tray_marker.type = Marker.CUBE
            tray_marker.action = Marker.ADD
            
            tray_marker.pose.position.x = center[0]
            tray_marker.pose.position.y = center[1]
            tray_marker.pose.position.z = 0.38  # Tray deck height
            
            # Calculate orientation from the tray's detected orientation vector
            # Orientation always represents the LONG side direction
            # scale.y (2.5m) should align with this orientation
            if 'orientation' in tray:
                import math
                orientation_vec = tray['orientation']
                
                # Calculate yaw angle from orientation vector (LONG side direction)
                # This will align scale.x with the LONG side direction
                yaw = math.atan2(orientation_vec[1], orientation_vec[0])
                
                # Convert yaw to quaternion (rotation around Z axis)
                tray_marker.pose.orientation.x = 0.0
                tray_marker.pose.orientation.y = 0.0
                tray_marker.pose.orientation.z = math.sin(yaw / 2.0)
                tray_marker.pose.orientation.w = math.cos(yaw / 2.0)
            else:
                tray_marker.pose.orientation.w = 1.0
            
            # Use tray dimensions from STL measurements
            import tray_config
            tray_marker.scale.x = tray_config.TRAY_FULL_LENGTH
            tray_marker.scale.y = tray_config.TRAY_FULL_WIDTH
            
            tray_marker.scale.z = 0.05  # Thin platform
            
            # Color (green for trays)
            tray_marker.color.r = 0.0
            tray_marker.color.g = 1.0
            tray_marker.color.b = 0.0
            tray_marker.color.a = 0.5
            tray_marker.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(tray_marker)
            
            # Tray text label
            tray_text = Marker()
            tray_text.header = odom_header
            tray_text.ns = "tray_labels"
            tray_text.id = marker_id
            marker_id += 1
            tray_text.type = Marker.TEXT_VIEW_FACING
            tray_text.action = Marker.ADD
            
            tray_text.pose.position.x = center[0]
            tray_text.pose.position.y = center[1]
            tray_text.pose.position.z = 0.6  # Above tray
            
            tray_text.text = f"TRAY #{i+1}\n{tray_type}\n{num_legs} legs"
            tray_text.scale.z = 0.25
            tray_text.color.r = 0.0
            tray_text.color.g = 1.0
            tray_text.color.b = 0.0
            tray_text.color.a = 1.0
            tray_text.lifetime = MARKER_LIFETIME
            
            marker_array.markers.append(tray_text)
        
        # Visualize detected leg pairs using create_leg_markers
        for i, leg_pair in enumerate(legs):
            pose = leg_pair['pose']
            pair_type = leg_pair['pair_type']
            transform = pose['transform']
            
            # Use create_leg_markers from leg_model to visualize the fitted template
            leg_markers, marker_id = create_leg_markers(
                transform,
                pair_type,
                odom_header,
                ns=f"fitted_legs_{i}",
                marker_id_start=marker_id
            )
            
            marker_array.markers.extend(leg_markers)

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
