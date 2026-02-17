#!/usr/bin/env python3

import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN


class TrayDetector:
    """Detects trays using edge detection at tray top + leg detection below

    Since the robot LiDAR is low-mounted and cannot see tray planes directly,
    this detector uses:
    1. Horizontal edge detection at tray top height (~0.38m)
    2. Leg cluster detection below tray (<0.30m)
    3. Validation of edge lengths and leg spacing against known dimensions

    ROBUST TO LOADED TRAYS:
    Height filtering ensures objects on top (>0.45m) don't interfere with detection.
    """

    # Tray dimensions (from tray_dimensions.yaml)
    TRAY_LENGTH = 5.1816  # meters (long side)
    TRAY_WIDTH = 2.4749   # meters (short side)
    TRAY_HEIGHT = 0.3794  # meters (vertical dimension)

    # Tray leg positions (inset from edges)
    LEG_X_INSET = 2.1717  # meters from long edge
    LEG_Y_INSET = 0.77    # meters from short edge

    # Calculated leg spacings
    LEG_SPACING_LONG = TRAY_LENGTH - 2 * LEG_X_INSET  # 0.8382m (between legs on short edge)
    LEG_SPACING_SHORT = TRAY_WIDTH - 2 * LEG_Y_INSET  # 0.9349m (between legs on long edge)

    def __init__(self,
                 voxel_size=0.03,
                 edge_height_min=0.30,
                 edge_height_max=0.45,
                 leg_height_min=0.0,
                 leg_height_max=0.30,
                 line_distance_threshold=0.05,
                 line_ransac_n=2,
                 line_num_iterations=1000,
                 length_tolerance=0.5,
                 width_tolerance=0.3,
                 leg_cluster_eps=0.15,
                 leg_cluster_min_samples=10,
                 leg_spacing_tolerance=0.3,
                 min_edge_points=50):
        """
        Initialize tray detector with edge + leg detection parameters

        Args:
            voxel_size: Voxel size for downsampling (m)
            edge_height_min: Minimum height for edge detection (m)
            edge_height_max: Maximum height for edge detection (m)
            leg_height_min: Minimum height for leg detection (m)
            leg_height_max: Maximum height for leg detection (m)
            line_distance_threshold: RANSAC distance threshold for line fitting (m)
            line_ransac_n: Number of points for line RANSAC
            line_num_iterations: RANSAC iterations for line fitting
            length_tolerance: Tolerance for matching tray length (m)
            width_tolerance: Tolerance for matching tray width (m)
            leg_cluster_eps: DBSCAN epsilon for leg clustering (m)
            leg_cluster_min_samples: DBSCAN minimum samples for leg clusters
            leg_spacing_tolerance: Tolerance for matching leg spacing (m)
            min_edge_points: Minimum points required for valid edge
        """
        self.voxel_size = voxel_size
        self.edge_height_min = edge_height_min
        self.edge_height_max = edge_height_max
        self.leg_height_min = leg_height_min
        self.leg_height_max = leg_height_max
        self.line_distance_threshold = line_distance_threshold
        self.line_ransac_n = line_ransac_n
        self.line_num_iterations = line_num_iterations
        self.length_tolerance = length_tolerance
        self.width_tolerance = width_tolerance
        self.leg_cluster_eps = leg_cluster_eps
        self.leg_cluster_min_samples = leg_cluster_min_samples
        self.leg_spacing_tolerance = leg_spacing_tolerance
        self.min_edge_points = min_edge_points

    def detect(self, points: np.ndarray) -> List[Dict]:
        """
        Detect trays using edge + leg detection

        Args:
            points: Nx3 numpy array of 3D points

        Returns:
            List of tray dictionaries with detection information
        """
        if len(points) == 0:
            return []

        print(f"\n[TrayDetector] === Starting detection with {len(points)} points ===")

        # Step 1: Detect edges at tray top height
        edges = self._detect_edges(points)
        print(f"[TrayDetector] Found {len(edges)} valid edges")

        # Step 2: Detect legs below tray
        legs = self._detect_legs(points)
        print(f"[TrayDetector] Found {len(legs)} leg clusters")

        # Step 3: Combine edge + leg information to detect trays
        trays = self._combine_edge_leg_detections(edges, legs)
        print(f"[TrayDetector] Detected {len(trays)} trays\n")

        return trays

    def _detect_edges(self, points: np.ndarray) -> List[Dict]:
        """
        Detect horizontal edges at tray top height using line fitting

        Returns:
            List of edge dictionaries with line parameters and dimensions
        """
        # Filter points at tray top height
        edge_mask = (points[:, 2] >= self.edge_height_min) & (points[:, 2] <= self.edge_height_max)
        edge_points = points[edge_mask]

        print(f"[TrayDetector] Edge detection: {len(edge_points)} points in height range [{self.edge_height_min}, {self.edge_height_max}]m")

        if len(edge_points) < self.min_edge_points:
            return []

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(edge_points)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        edges = []
        remaining_pcd = pcd
        max_iterations = 5

        for iteration in range(max_iterations):
            if len(remaining_pcd.points) < self.min_edge_points:
                break

            # Fit 3D line using RANSAC
            edge_info = self._fit_line_ransac(np.asarray(remaining_pcd.points))

            if edge_info is not None:
                print(f"[TrayDetector]   Edge {iteration}: length={edge_info['length']:.2f}m, type={edge_info['edge_type']}, points={edge_info['num_points']}")
                edges.append(edge_info)

                # Remove inlier points
                inlier_mask = np.ones(len(remaining_pcd.points), dtype=bool)
                inlier_mask[edge_info['inlier_indices']] = False
                remaining_pcd = remaining_pcd.select_by_index(np.where(inlier_mask)[0])
            else:
                break

        return edges

    def _fit_line_ransac(self, points: np.ndarray) -> Optional[Dict]:
        """
        Fit a 3D line to points using RANSAC and validate dimensions

        Returns:
            Dictionary with line parameters if valid, None otherwise
        """
        if len(points) < self.min_edge_points:
            return None

        best_inliers = []
        best_line_point = None
        best_line_direction = None
        best_score = 0

        # RANSAC iterations
        for _ in range(self.line_num_iterations):
            # Sample 2 random points
            if len(points) < 2:
                break

            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            # Line direction
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)

            if direction_norm < 0.01:
                continue

            direction = direction / direction_norm

            # Find inliers
            inliers = []
            for i, pt in enumerate(points):
                # Distance from point to line
                vec = pt - p1
                cross = np.cross(vec, direction)
                distance = np.linalg.norm(cross)

                if distance < self.line_distance_threshold:
                    inliers.append(i)

            # Update best fit
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
                best_line_point = p1
                best_line_direction = direction

        # Validate best fit
        if len(best_inliers) < self.min_edge_points:
            return None

        # Get inlier points
        inlier_points = points[best_inliers]

        # Project points onto line to find endpoints
        projections = []
        for pt in inlier_points:
            vec = pt - best_line_point
            t = np.dot(vec, best_line_direction)
            projections.append(t)

        t_min, t_max = min(projections), max(projections)
        line_length = t_max - t_min

        # Calculate line endpoints
        start_point = best_line_point + t_min * best_line_direction
        end_point = best_line_point + t_max * best_line_direction
        center = (start_point + end_point) / 2.0

        # Determine edge type based on length
        edge_type = 'unknown'
        confidence = 0.0

        if abs(line_length - self.TRAY_LENGTH) < self.length_tolerance:
            edge_type = 'long'
            confidence = 1.0 - abs(line_length - self.TRAY_LENGTH) / self.length_tolerance
        elif abs(line_length - self.TRAY_WIDTH) < self.width_tolerance:
            edge_type = 'short'
            confidence = 1.0 - abs(line_length - self.TRAY_WIDTH) / self.width_tolerance
        else:
            # Check if it's a partial edge (at least 1.5m long)
            if line_length > 1.5:
                edge_type = 'partial'
                confidence = 0.4
            else:
                return None  # Too short to be a tray edge

        return {
            'start': start_point,
            'end': end_point,
            'center': center,
            'direction': best_line_direction,
            'length': line_length,
            'edge_type': edge_type,
            'confidence': confidence,
            'num_points': len(best_inliers),
            'inlier_indices': best_inliers
        }

    def _detect_legs(self, points: np.ndarray) -> List[Dict]:
        """
        Detect tray legs using clustering on points below tray deck

        Returns:
            List of leg cluster dictionaries
        """
        # Filter points at leg height (below tray deck)
        leg_mask = (points[:, 2] >= self.leg_height_min) & (points[:, 2] <= self.leg_height_max)
        leg_points = points[leg_mask]

        print(f"[TrayDetector] Leg detection: {len(leg_points)} points in height range [{self.leg_height_min}, {self.leg_height_max}]m")

        if len(leg_points) < self.leg_cluster_min_samples:
            return []

        # Use DBSCAN clustering to find individual legs
        clustering = DBSCAN(eps=self.leg_cluster_eps, min_samples=self.leg_cluster_min_samples)
        labels = clustering.fit_predict(leg_points[:, :2])  # Only use XY for clustering

        legs = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        print(f"[TrayDetector]   Found {len(unique_labels)} clusters")

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = leg_points[cluster_mask]

            # Calculate cluster center (XY only, use average Z)
            center_xy = np.mean(cluster_points[:, :2], axis=0)
            avg_z = np.mean(cluster_points[:, 2])
            center = np.array([center_xy[0], center_xy[1], avg_z])

            legs.append({
                'center': center,
                'num_points': len(cluster_points),
                'points': cluster_points
            })

        return legs

    def _combine_edge_leg_detections(self, edges: List[Dict], legs: List[Dict]) -> List[Dict]:
        """
        Combine edge and leg detections to identify trays

        Strategy:
        1. If we find valid edge(s) matching tray dimensions, check for supporting legs
        2. If we find a corner (2 perpendicular edges), validate with legs or edge dimensions
        3. Compute full tray pose from detected features

        Returns:
            List of validated tray detections
        """
        trays = []

        # Case 1: Corner detection (two perpendicular edges)
        corners = self._find_corners(edges)
        for corner in corners:
            tray = self._validate_corner_detection(corner, legs)
            if tray is not None:
                trays.append(tray)

        # Case 2: Single edge detection validated by legs
        used_edges = set()
        for corner in corners:
            used_edges.add(id(corner['edge1']))
            used_edges.add(id(corner['edge2']))

        for edge in edges:
            if id(edge) in used_edges:
                continue

            # Only consider edges with known type (long/short)
            if edge['edge_type'] in ['long', 'short']:
                tray = self._validate_single_edge(edge, legs)
                if tray is not None:
                    trays.append(tray)

        return trays

    def _find_corners(self, edges: List[Dict]) -> List[Dict]:
        """
        Find corners from perpendicular edge pairs

        Returns:
            List of corner dictionaries with edge pairs
        """
        corners = []

        for i, edge1 in enumerate(edges):
            for edge2 in edges[i+1:]:
                # Check if edges are approximately perpendicular
                dot_product = abs(np.dot(edge1['direction'], edge2['direction']))

                # Perpendicular if dot product close to 0
                if dot_product < 0.3:  # Approx perpendicular (angle > 72 degrees)
                    # Check if edges are close enough to form a corner
                    dist = np.linalg.norm(edge1['center'] - edge2['center'])

                    if dist < 3.0:  # Within reasonable distance
                        print(f"[TrayDetector] Found corner: {edge1['edge_type']} + {edge2['edge_type']}, dist={dist:.2f}m")
                        corners.append({
                            'edge1': edge1,
                            'edge2': edge2,
                            'distance': dist
                        })

        return corners

    def _validate_corner_detection(self, corner: Dict, legs: List[Dict]) -> Optional[Dict]:
        """
        Validate corner detection and compute tray pose

        Args:
            corner: Corner dictionary with two edges
            legs: List of detected legs

        Returns:
            Tray detection dictionary if valid, None otherwise
        """
        edge1 = corner['edge1']
        edge2 = corner['edge2']

        # Check if we have both long and short edges
        has_long = edge1['edge_type'] == 'long' or edge2['edge_type'] == 'long'
        has_short = edge1['edge_type'] == 'short' or edge2['edge_type'] == 'short'

        if has_long and has_short:
            # Perfect corner with both sides
            print(f"[TrayDetector] ✓ Valid corner with long + short edges")

            # Determine which is which
            long_edge = edge1 if edge1['edge_type'] == 'long' else edge2
            short_edge = edge1 if edge1['edge_type'] == 'short' else edge2

            # Estimate tray center from corner
            # Corner is at intersection of edges
            corner_point = (edge1['center'] + edge2['center']) / 2.0

            # Tray center is offset by half dimensions from corner
            long_dir = long_edge['direction']
            short_dir = short_edge['direction']

            # Ensure directions point inward to tray
            # (this is approximate - might need refinement based on actual geometry)
            tray_center = corner_point + long_dir * (self.TRAY_LENGTH / 4.0) + short_dir * (self.TRAY_WIDTH / 4.0)

            confidence = (edge1['confidence'] + edge2['confidence']) / 2.0

            # Check for supporting legs if available
            if len(legs) >= 2:
                leg_support = self._check_leg_support(tray_center, legs)
                confidence *= (0.7 + 0.3 * leg_support)

            return {
                'center': tray_center,
                'type': 'corner',
                'edge1': edge1,
                'edge2': edge2,
                'confidence': confidence,
                'num_points': edge1['num_points'] + edge2['num_points'],
                'side_type': 'corner',
                'dimensions': {
                    'width': short_edge['length'],
                    'height': self.TRAY_HEIGHT
                },
                'bounds': self._compute_bounds(tray_center, edge1, edge2)
            }

        # Partial corner - check legs
        elif len(legs) >= 2:
            leg_support = self._check_leg_support(corner['edge1']['center'], legs)
            if leg_support > 0.5:
                print(f"[TrayDetector] ✓ Partial corner validated by legs (support={leg_support:.2f})")

                tray_center = (edge1['center'] + edge2['center']) / 2.0
                confidence = (edge1['confidence'] + edge2['confidence']) / 2.0 * leg_support

                return {
                    'center': tray_center,
                    'type': 'partial_corner',
                    'edge1': edge1,
                    'edge2': edge2,
                    'confidence': confidence,
                    'num_points': edge1['num_points'] + edge2['num_points'],
                    'side_type': 'partial',
                    'dimensions': {
                        'width': max(edge1['length'], edge2['length']),
                        'height': self.TRAY_HEIGHT
                    },
                    'bounds': self._compute_bounds(tray_center, edge1, edge2)
                }

        return None

    def _validate_single_edge(self, edge: Dict, legs: List[Dict]) -> Optional[Dict]:
        """
        Validate single edge detection using leg support

        Args:
            edge: Edge dictionary
            legs: List of detected legs

        Returns:
            Tray detection dictionary if valid, None otherwise
        """
        if len(legs) < 2:
            return None

        # Check if legs support this edge
        leg_support = self._check_leg_support(edge['center'], legs)

        if leg_support > 0.5:
            print(f"[TrayDetector] ✓ Single {edge['edge_type']} edge validated by legs (support={leg_support:.2f})")

            confidence = edge['confidence'] * leg_support

            # Estimate tray center (perpendicular to edge)
            # For single edge, we don't know exact position, use edge center
            tray_center = edge['center'].copy()

            return {
                'center': tray_center,
                'type': 'single_edge',
                'edge': edge,
                'confidence': confidence,
                'num_points': edge['num_points'],
                'side_type': edge['edge_type'],
                'dimensions': {
                    'width': edge['length'],
                    'height': self.TRAY_HEIGHT
                },
                'bounds': self._compute_bounds_single_edge(edge)
            }

        return None

    def _check_leg_support(self, reference_point: np.ndarray, legs: List[Dict]) -> float:
        """
        Check if detected legs support a tray at given position

        Returns:
            Support score 0-1 (1 = perfect match)
        """
        if len(legs) < 2:
            return 0.0

        # Check leg spacing
        leg_positions = [leg['center'][:2] for leg in legs]  # XY only

        # Try to find leg pairs with expected spacing
        max_score = 0.0

        for i, leg1_pos in enumerate(leg_positions):
            for leg2_pos in leg_positions[i+1:]:
                spacing = np.linalg.norm(leg1_pos - leg2_pos)

                # Check if spacing matches long or short leg spacing
                long_match = abs(spacing - self.LEG_SPACING_LONG) < self.leg_spacing_tolerance
                short_match = abs(spacing - self.LEG_SPACING_SHORT) < self.leg_spacing_tolerance

                if long_match or short_match:
                    # Found matching leg pair
                    if long_match:
                        score = 1.0 - abs(spacing - self.LEG_SPACING_LONG) / self.leg_spacing_tolerance
                    else:
                        score = 1.0 - abs(spacing - self.LEG_SPACING_SHORT) / self.leg_spacing_tolerance

                    max_score = max(max_score, score)

        return max_score

    def _compute_bounds(self, center: np.ndarray, edge1: Dict, edge2: Dict) -> Dict:
        """Compute bounding box from corner detection"""
        all_points = np.vstack([
            [edge1['start'], edge1['end']],
            [edge2['start'], edge2['end']]
        ])

        min_pt = np.min(all_points, axis=0)
        max_pt = np.max(all_points, axis=0)

        return {'min': min_pt, 'max': max_pt}

    def _compute_bounds_single_edge(self, edge: Dict) -> Dict:
        """Compute bounding box from single edge"""
        min_pt = np.minimum(edge['start'], edge['end'])
        max_pt = np.maximum(edge['start'], edge['end'])

        # Expand perpendicular to edge (estimate)
        if edge['edge_type'] == 'long':
            expand = self.TRAY_WIDTH / 2.0
        elif edge['edge_type'] == 'short':
            expand = self.TRAY_LENGTH / 2.0
        else:
            expand = 1.0

        # Expand in all directions (crude approximation)
        min_pt -= expand
        max_pt += expand

        return {'min': min_pt, 'max': max_pt}
