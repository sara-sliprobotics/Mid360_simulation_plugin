import numpy as np
import math
from leg_detector import LegDetector
import tray_config


# This is strategy prioritize the Leg Pattern first to eliminate 90% of false positives (walls, random boxes) before you even look for the edge.

#    The Strategy: "Leg-First" Detection

#     Find "Leg Candidates": Slice low (0.05−0.15m), flatten, cluster, and filter for Small Objects Only (<20cm).

#     Pattern Match (The "Fit"):

#         If >2 Legs: Do they form a rectangle of size 2.47m×0.93m (Long Side) or 2.47m×0.83m (Short Side)?

#         If 2 Legs: Are they exactly 0.93m (Long Side spacing) or 0.83m (Short Side spacing) apart?

#     Edge Verification (The "Proof"):

#         Look Above the confirmed legs (0.30−0.45m).

#         Is there a Line connecting them?

#         Yes: → CONFIRMED TRAY.

#     To elimate false alarms, we use Convex Corner (a tray sticking out) and a Concave Corner (the corner of a room).

#     Tray Corner: The corner "points" at the robot. (Robot is Outside).

#     Room Corner: The corner "points" away from the robot. (Robot is Inside).

# Here is the logic to add to your detector.
# The Logic: "The Arrow Test"

#     Identify the Corner: If you have 3 legs, one is the "Middle" (Corner) leg, and the other two are the "Ends".

#     Create Edge Vectors: Calculate vectors from the Corner to the Ends (V1​ and V2​).

#     Calculate "Corner Direction": The corner points in the direction of −(V1​+V2​).

#     Check Robot: Is the robot standing in that direction?

#         Dot Product > 0: Corner faces Robot → VALID TRAY.

#         Dot Product < 0: Corner faces away → WALL (Ignore).

class LegFirstTrayDetector:
    EDGE_Z_MIN = tray_config.EDGE_Z_MIN
    EDGE_Z_MAX = tray_config.EDGE_Z_MAX
    SPACING_SHORT = tray_config.SPACING_SHORT
    SPACING_LONG = tray_config.SPACING_LONG
    SPACING_TOL = tray_config.SPACING_TOL
    TRAY_FULL_LENGTH = tray_config.TRAY_FULL_LENGTH
    TRAY_FULL_WIDTH = tray_config.TRAY_FULL_WIDTH

    def __init__(self):
        self.leg_detector = LegDetector()

    def detect(self, pcd):
        detected_trays = []

        # Debug: Check point cloud size
        total_points = len(np.asarray(pcd.points))
        # print(f"\n[TRAY DETECTOR] Total points in cloud: {total_points}")

        # --- PHASE 1: FIND LEG CANDIDATES ---
        # Returns list of {center, size, pcd}
        legs = self.leg_detector.find_candidates(pcd)
        
        # print(f"[TRAY DETECTOR] Found {len(legs)} leg candidates")
        # for i, leg in enumerate(legs):
        #     print(f"  Leg {i}: center={leg['center']}, points={len(np.asarray(leg['pcd'].points))}")
        
        if len(legs) < 2:
            # print("  [INFO] Less than 2 legs found. Aborting.")
            return []

        # --- PHASE 2: FIND PAIRS ---
        pairs = self.leg_detector.find_pairs(legs)

        # --- PHASE 3: PROCESS CORNERS (3 LEGS) ---
        # If two pairs share a leg, they form a corner.
        # Track used leg indices so each leg can only belong to one detected tray,
        # preventing duplicate detections when all 4 legs are visible.
        used_leg_indices = set()

        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i >= j: continue

                # Check intersection (do they share exactly 1 leg?)
                shared = pair1['indices'].intersection(pair2['indices'])

                if len(shared) == 1:
                    all_leg_indices = pair1['indices'] | pair2['indices']

                    # Skip if any of the 3 legs already belongs to a detected tray
                    if all_leg_indices & used_leg_indices:
                        continue

                    # FOUND POTENTIAL CORNER!
                    shared_idx = list(shared)[0]
                    corner_leg = legs[shared_idx]

                    # Identify the "End Legs"
                    idx1 = list(pair1['indices'] - shared)[0]
                    idx2 = list(pair2['indices'] - shared)[0]
                    end_leg1 = legs[idx1]
                    end_leg2 = legs[idx2]

                    # *** CRITICAL CHECK: IS IT FACING THE ROBOT? ***
                    if self._check_corner_facing(corner_leg, end_leg1, end_leg2):
                        # print("  ✅ Valid Convex Corner (Facing Robot)")

                        # Verify Edge above BOTH sides
                        edge_cloud = self.leg_detector.get_flattened_layer(pcd, self.EDGE_Z_MIN, self.EDGE_Z_MAX)
                        valid_1 = self._verify_edge_above(corner_leg, end_leg1, edge_cloud)
                        valid_2 = self._verify_edge_above(corner_leg, end_leg2, edge_cloud)

                        if valid_1 and valid_2:
                            c = corner_leg['center']
                            e1 = end_leg1['center']
                            e2 = end_leg2['center']

                            # Estimate the 4th leg and compute tray center
                            fourth_leg = c + (e1 - c) + (e2 - c)
                            tray_center = (c + e1 + e2 + fourth_leg) / 4.0
                            tray_center[2] = self.EDGE_Z_MIN + (self.EDGE_Z_MAX - self.EDGE_Z_MIN) / 2

                            side1_vec = e1 - c

                            detected_trays.append({
                                "type": "TRAY_CORNER",
                                "legs": [corner_leg, end_leg1, end_leg2],
                                "center": tray_center,
                                "corner_leg": c,
                                "orientation": side1_vec
                            })
                            used_leg_indices |= all_leg_indices
                    # else:
                        # print("  ❌ Ignored Concave Corner (Wall)")

        # --- PHASE 4: PROCESS REMAINING PAIRS (2 LEGS) ---
        # If a pair wasn't used in a corner, check it as a standalone side
        edge_cloud = self.leg_detector.get_flattened_layer(pcd, self.EDGE_Z_MIN, self.EDGE_Z_MAX)
        # print(f"[TRAY DETECTOR] Edge layer ({self.EDGE_Z_MIN}m to {self.EDGE_Z_MAX}m) has {len(edge_cloud.points)} points")

        for i, pair in enumerate(pairs):
            if pair['indices'] & used_leg_indices: continue
            
            # print(f"[TRAY DETECTOR] Checking pair {i}: {pair['type']} side, dist={pair['dist']:.3f}m")
            
            # Verify Edge above this pair
            if self._verify_edge_above(pair['legs'][0], pair['legs'][1], edge_cloud):
                # Calculate tray center for 2-leg detection
                leg1_center = pair['legs'][0]['center']
                leg2_center = pair['legs'][1]['center']
                
                # print(f"  [CENTER DEBUG] Leg 1: {leg1_center[:2]}")
                # print(f"  [CENTER DEBUG] Leg 2: {leg2_center[:2]}")
                
                # Midpoint between the two detected legs
                midpoint = (leg1_center + leg2_center) / 2.0
                # print(f"  [CENTER DEBUG] Midpoint: {midpoint[:2]}")
                
                # Vector along the detected side
                side_vector = leg2_center - leg1_center
                side_length = np.linalg.norm(side_vector[:2])
                side_dir = side_vector[:2] / side_length
                
                # Perpendicular direction (rotate 90 degrees)
                perp_dir_1 = np.array([-side_dir[1], side_dir[0]])  # Counter-clockwise
                perp_dir_2 = np.array([side_dir[1], -side_dir[0]])  # Clockwise
                
                # Determine which perpendicular direction faces away from robot
                midpoint_2d = midpoint[:2]
                to_robot = -midpoint_2d  # Vector from midpoint to robot at (0,0)
                
                # Choose the perpendicular direction that points away from robot
                if np.dot(perp_dir_1, to_robot) < np.dot(perp_dir_2, to_robot):
                    perp_dir = perp_dir_1  # This one points away from robot
                else:
                    perp_dir = perp_dir_2
                
                # print(f"  [CENTER DEBUG] Perp direction (away from robot): {perp_dir}")
                # print(f"  [CENTER DEBUG] Dot products: perp1={np.dot(perp_dir_1, to_robot):.3f}, perp2={np.dot(perp_dir_2, to_robot):.3f}")
                
                # Calculate tray center by shifting from midpoint using leg spacing
                if pair['type'] == 'SHORT':
                    # Detected short side, shift by half of LONG leg spacing (perpendicular distance)
                    shift_distance = self.SPACING_LONG / 2.0
                else:  # LONG
                    # Detected long side, shift by half of SHORT leg spacing (perpendicular distance)
                    shift_distance = self.SPACING_SHORT / 2.0
                
                # print(f"  [CENTER DEBUG] Shift distance: {shift_distance:.3f}m")
                
                # Tray center is midpoint shifted in perpendicular direction
                tray_center = midpoint.copy()
                tray_center[:2] = midpoint[:2] + perp_dir * shift_distance
                tray_center[2] = self.EDGE_Z_MIN + (self.EDGE_Z_MAX - self.EDGE_Z_MIN) / 2
                
                # print(f"  [CENTER DEBUG] Final tray center: {tray_center[:2]}")
                
                # Orientation should always be along the LONG side
                if pair['type'] == 'SHORT':
                    # We detected short side, so orientation is perpendicular (long side direction)
                    long_side_vector = np.array([perp_dir[0], perp_dir[1], 0.0])
                else:  # LONG
                    # We detected long side, so orientation is along detected side
                    long_side_vector = side_vector
                
                # print(f"  [CENTER DEBUG] Orientation (LONG side): {long_side_vector[:2]}")
                
                detected_trays.append({
                    "type": f"TRAY_SIDE_{pair['type']}",
                    "legs": pair['legs'],
                    "center": tray_center,
                    "orientation": long_side_vector,
                    "side_length": pair['dist']
                })
                used_leg_indices |= pair['indices']
                # print(f"  ✅ Confirmed 2-Leg Tray Side ({pair['type']}) - Center: {tray_center[:2]}")
            else:
                pass
                # print(f"  ✗ Failed edge verification for {pair['type']} side")

        return detected_trays

    def _check_corner_facing(self, corner_leg, end1, end2):
        """
        Dot Product Check: Does the corner point at the robot (0,0)?
        """
        c = corner_leg['center'][:2]
        e1 = end1['center'][:2]
        e2 = end2['center'][:2]
        
        # Vectors from Corner -> Ends
        v1 = e1 - c
        v2 = e2 - c
        
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        
        # Corner Bisector (Points OUT of the corner)
        # We reverse (v1+v2) because v1/v2 point "into" the structure
        corner_vector = -(v1 + v2)
        
        # Vector from Corner to Robot (Robot is at 0,0)
        to_robot = np.array([0,0]) - c
        to_robot = to_robot / (np.linalg.norm(to_robot) + 1e-6)
        
        # Dot Product
        # > 0: Corner faces robot (Convex) -> TRAY
        # < 0: Corner faces away (Concave) -> WALL
        score = np.dot(corner_vector, to_robot)
        
        return score > 0.2 # Use 0.2 buffer to be safe

    def _verify_edge_above(self, leg1, leg2, edge_cloud):
        """
        Checks if edge points connect these two legs
        """
        total_edge_points = len(edge_cloud.points)
        # print(f"  [EDGE VERIFY] Total edge cloud points: {total_edge_points}")
        
        if total_edge_points < 5:
            # print(f"  [EDGE VERIFY] ✗ Too few edge points ({total_edge_points} < 5)")
            return False
        
        p1 = leg1['center'][:2]
        p2 = leg2['center'][:2]
        
        # Calculate line direction and length
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)
        # print(f"  [EDGE VERIFY] Leg distance: {line_length:.3f}m")
        
        if line_length < 0.1:
            # print(f"  [EDGE VERIFY] ✗ Legs too close")
            return False
        
        line_dir = line_vec / line_length
        
        # Get edge points
        edge_pts = np.asarray(edge_cloud.points)
        if len(edge_pts) == 0:
            return False
        
        # Filter points near the line segment
        edge_pts_2d = edge_pts[:, :2]
        
        # Calculate perpendicular distance from each point to the line
        # Vector from p1 to each point
        vecs_to_pts = edge_pts_2d - p1
        
        # Project onto line direction
        proj_lengths = np.dot(vecs_to_pts, line_dir)
        
        # Filter points that are within the line segment bounds (with some tolerance)
        tolerance = 0.1  # 10cm tolerance beyond legs
        valid_proj = (proj_lengths >= -tolerance) & (proj_lengths <= line_length + tolerance)
        
        # Calculate perpendicular distance to line
        proj_vecs = np.outer(proj_lengths, line_dir)
        perp_vecs = vecs_to_pts - proj_vecs
        perp_dists = np.linalg.norm(perp_vecs, axis=1)
        
        # Debug: Show distribution of perpendicular distances
        # if len(perp_dists[valid_proj]) > 0:
        #     print(f"  [EDGE VERIFY] Perp distances: min={perp_dists[valid_proj].min():.3f}m, max={perp_dists[valid_proj].max():.3f}m, median={np.median(perp_dists[valid_proj]):.3f}m")
        
        # Points within perpendicular distance and along the line segment
        # Increased tolerance since edge may be offset from leg centers
        max_perp_dist = 0.30  # 30cm to account for edge offset from legs
        valid_points = valid_proj & (perp_dists < max_perp_dist)
        
        num_edge_points = np.sum(valid_points)
        
        # print(f"  [EDGE VERIFY] Valid edge points: {num_edge_points}")
        # print(f"  [EDGE VERIFY] Max perp dist: {max_perp_dist}m, Tolerance: {tolerance}m")
        
        # Require at least 20 points to confirm edge
        min_required_points = 20
        success = num_edge_points >= min_required_points
        
        # if success:
        #     print(f"  [EDGE VERIFY] ✓ Edge confirmed ({num_edge_points} >= {min_required_points} points)")
        # else:
        #     print(f"  [EDGE VERIFY] ✗ Not enough edge points ({num_edge_points} < {min_required_points})")
        
        return success