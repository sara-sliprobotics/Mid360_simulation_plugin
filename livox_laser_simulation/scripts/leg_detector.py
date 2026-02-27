import numpy as np
import open3d as o3d
from frame_accumulator import FrameAccumulator
from leg_model import fit_live_lidar_to_tray
import tray_config


class LegDetector:
    LEG_Z_MIN = tray_config.LEG_Z_MIN
    LEG_Z_MAX = tray_config.LEG_Z_MAX
    MAX_LEG_SIZE = tray_config.MAX_LEG_SIZE
    MIN_LEG_SIZE = tray_config.MIN_LEG_SIZE
    MIN_LEG_POINTS = 15       # Reject clusters with fewer points (noise)

    def __init__(self, accumulator: FrameAccumulator = None):
        self.accumulator = accumulator

    def find_candidates(self, pcd, keep_3d=False):
        """
        Slice -> Flatten (for clustering) -> Cluster -> Size Filter
        Returns list of {center, pcd} dicts for each leg candidate.

        Args:
            pcd: Input point cloud
            keep_3d: If True, also returns 'pcd_3d' with original Z values for ICP.
                     If False (default), only returns flattened pcd (faster).
        """
        if keep_3d:
            # Slice to Z range keeping 3D, then flatten a copy for clustering
            leg_3d_pcd = self._get_z_slice(pcd, self.LEG_Z_MIN, self.LEG_Z_MAX)
            leg_layer_points = len(leg_3d_pcd.points)
            pts_3d = np.asarray(leg_3d_pcd.points)
            pts_flat = pts_3d.copy()
            pts_flat[:, 2] = 0
            flat_pcd = o3d.geometry.PointCloud()
            flat_pcd.points = o3d.utility.Vector3dVector(pts_flat)
        else:
            flat_pcd = self.get_flattened_layer(pcd, self.LEG_Z_MIN, self.LEG_Z_MAX)
            leg_layer_points = len(flat_pcd.points)

        print(f"[LEG DETECTOR] Points in leg layer ({self.LEG_Z_MIN}m to {self.LEG_Z_MAX}m): {leg_layer_points}")

        if leg_layer_points < 5:
            print("  [WARN] Too few points in leg layer")
            return []

        # Cluster on flattened data (DBSCAN works better in 2D)
        labels = np.array(flat_pcd.cluster_dbscan(eps=0.05, min_points=5))

        candidates = []
        if len(labels) == 0:
            return candidates

        for i in range(labels.max() + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_flat = flat_pcd.select_by_index(cluster_indices)

            try:
                aabb = cluster_flat.get_axis_aligned_bounding_box()
                extent = aabb.get_extent()
                length = max(extent[0], extent[1])
                center = aabb.get_center()

                if self.MIN_LEG_SIZE < length < self.MAX_LEG_SIZE:
                    candidate = {
                        "center": center,
                        "pcd": cluster_flat
                    }
                    if keep_3d:
                        candidate["pcd_3d"] = leg_3d_pcd.select_by_index(cluster_indices)
                    candidates.append(candidate)
            except RuntimeError:
                continue

        # Filter by minimum point count (removes noise clusters)
        candidates = [c for c in candidates if len(c['pcd'].points) >= self.MIN_LEG_POINTS]

        return candidates

    def _get_z_slice(self, pcd, z_min, z_max):
        """Crop to Z slice, preserving original 3D coordinates."""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-100, -100, z_min]),
            max_bound=np.array([ 100,  100, z_max])
        )
        return pcd.crop(bbox)

    def detect(self, pcd=None, box=None, enter_side="SHORT"):
        """
        Full leg detection pipeline:
          1. Get cloud from accumulator if set, otherwise fall back to the provided pcd.
          2. If a bounding box is provided, crop the cloud to it before searching.
          3. Find size-filtered leg candidates from the (cropped) cloud.
          4. Confirm candidates by checking they form valid pairs at the expected spacing.
          5. Filter pairs to only the requested side (enter_side).
          6. Fit a leg model to each confirmed pair for an accurate pose.

        Args:
            pcd:           Optional single-frame fallback when no accumulator is set.
            box:           Optional o3d.geometry.AxisAlignedBoundingBox to restrict the
                           search region. If None the full cloud is used.
            enter_side:    "SHORT" or "LONG" — only pairs of this type are processed.

        Returns a list of dicts: {center, pose, pcd}
        Returns [] if no cloud is available.
        """
        # Step 1: Get the cloud to run detection on
        if self.accumulator is not None:
            cloud = self.accumulator.get_accumulated_cloud()
            if cloud is None:
                print("[LEG DETECTOR] Accumulator is empty, no frames received yet")
                return []
            acc_pts = np.asarray(cloud.points)
            print(f"[LEG DETECTOR DEBUG] Accumulated cloud: {len(acc_pts)} pts, "
                  f"Z range: [{acc_pts[:,2].min():.3f}, {acc_pts[:,2].max():.3f}], "
                  f"X range: [{acc_pts[:,0].min():.3f}, {acc_pts[:,0].max():.3f}], "
                  f"Y range: [{acc_pts[:,1].min():.3f}, {acc_pts[:,1].max():.3f}]")
            # Count points in leg layer BEFORE cropping
            leg_mask = (acc_pts[:,2] >= self.LEG_Z_MIN) & (acc_pts[:,2] <= self.LEG_Z_MAX)
            print(f"[LEG DETECTOR DEBUG] Points in leg Z layer [{self.LEG_Z_MIN}, {self.LEG_Z_MAX}] "
                  f"BEFORE box crop: {leg_mask.sum()}")
        elif pcd is not None:
            cloud = pcd
            pcd_pts = np.asarray(cloud.points)
            print(f"[LEG DETECTOR DEBUG] Single-frame cloud: {len(pcd_pts)} pts, "
                  f"Z range: [{pcd_pts[:,2].min():.3f}, {pcd_pts[:,2].max():.3f}]")
        else:
            print("[LEG DETECTOR] No accumulator set and no pcd provided")
            return []

        # Step 2: Crop to bounding box if one was provided
        if box is not None:
            pre_crop = len(cloud.points)
            cloud = cloud.crop(box)
            print(f"[LEG DETECTOR DEBUG] Box crop: {pre_crop} -> {len(cloud.points)} pts")
            if len(cloud.points) > 0:
                cropped_pts = np.asarray(cloud.points)
                print(f"[LEG DETECTOR DEBUG] After crop Z range: [{cropped_pts[:,2].min():.3f}, {cropped_pts[:,2].max():.3f}]")
            if len(cloud.points) == 0:
                print(f"[LEG DETECTOR] No points inside the provided box "
                      f"(min={np.asarray(box.min_bound)}, max={np.asarray(box.max_bound)})")
                return []

        # Step 3: Find candidates with 3D data preserved for Point-to-Plane ICP
        candidates = self.find_candidates(cloud, keep_3d=True)
        if len(candidates) < 2:
            return []

        # Step 4: Find pairs at the expected spacing to confirm which candidates
        # are real legs (anything not in a valid pair is likely noise/clutter)
        pairs = self.find_pairs(candidates)
        if not pairs:
            return []

        # Filter pairs to only the requested side
        matching_pairs = [p for p in pairs if p['type'] == enter_side]
        # print(f"[LEG DETECTOR] {len(matching_pairs)} {enter_side} pairs from {len(pairs)} total pairs")

        if not matching_pairs:
            print(f"[LEG DETECTOR] No {enter_side} pairs found")
            return []

        # Sort pairs by how closely their distance matches the expected spacing,
        # so the best-fitting pairs get processed first.
        matching_pairs.sort(key=lambda p: abs(p['dist'] - (
            tray_config.SPACING_SHORT if p['type'] == 'SHORT' else tray_config.SPACING_LONG)))

        # Fit leg model to each matching pair.
        # Track used leg indices so each leg is only used once,
        # preventing redundant cross-diagonal pairs from being ICP-fitted.
        # Also track detected centers to skip redundant pairs from the same tray
        # (e.g. both SHORT sides produce nearly identical centers).
        detected_legs = []
        used_legs = set()
        detected_centers = []

        for pair in matching_pairs:
            idx_set = pair['indices']
            # Skip if either leg was already used in a successfully fitted pair
            if idx_set & used_legs:
                continue

            # Skip if this pair's midpoint is too close to an already-detected center
            # (means it's the other side of the same tray — redundant)
            pair_mid = (pair['legs'][0]['center'][:2] + pair['legs'][1]['center'][:2]) / 2.0
            too_close = False
            for dc in detected_centers:
                if np.linalg.norm(pair_mid - dc) < 1.0:
                    too_close = True
                    break
            if too_close:
                continue

            # Get the two legs in this pair
            leg1, leg2 = pair['legs']
            pair_type = pair['type']

            # Fit the appropriate template (short or long side)
            pose = self._fit_leg_model_pair(leg1, leg2, pair_type)
            
            if pose is not None:
                detected_legs.append({
                    "pair_type": pair_type,
                    "center": pose['center'],
                    "pose": pose,
                    "legs": [leg1, leg2]
                })
                used_legs |= idx_set
                detected_centers.append(pose['center'][:2])
                print(f"  ✅ Confirmed 2-Leg Tray Side ({pair_type}) - Center: {pose['center']}")

        return detected_legs

    def _fit_leg_model_pair(self, leg1, leg2, pair_type, is_under_tray=False):
        """
        Fit the leg model template to a pair of detected legs using Point-to-Plane ICP.
        Uses the 3D point clouds (pcd_3d) so normals are meaningful.
        The ICP result is sanitised to XY + yaw only; the full 3D template
        is used for visualisation.

        Args:
            leg1: First leg dict with 'center', 'pcd' (flat), and 'pcd_3d' (3D)
            leg2: Second leg dict with 'center', 'pcd' (flat), and 'pcd_3d' (3D)
            pair_type: "SHORT" or "LONG" indicating which template to use
            is_under_tray: If True, flips the long-side template 180 deg

        Returns:
            pose: dict with 'center', 'x', 'y', 'yaw', and 'transform' matrix
                  or None if fitting failed
        """
        try:
            # Build ICP inputs using the 3D clouds for meaningful normals
            icp_leg1 = {'center': leg1['center'], 'pcd': leg1['pcd_3d']}
            icp_leg2 = {'center': leg2['center'], 'pcd': leg2['pcd_3d']}

            final_x, final_y, final_yaw, transform = fit_live_lidar_to_tray(
                icp_leg1,
                icp_leg2,
                side=pair_type,
                is_under_tray=is_under_tray
            )
            
            return {
                'center': np.array([final_x, final_y]),
                'x': final_x,
                'y': final_y,
                'yaw': final_yaw,
                'transform': transform
            }
        except Exception as e:
            print(f"  [WARN] ICP fitting failed for {pair_type} pair: {e}")
            return None

    def find_pairs(self, legs):
        """
        Find valid pairs of legs whose XY distance matches either the long or
        short spacing of the target object (within tolerance).

        Returns a list of dicts:
            {type: "LONG"|"SHORT", legs: [leg1, leg2], indices: {i, j}, dist: float}
        """
        pairs = []
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                leg1 = legs[i]
                leg2 = legs[j]

                dist = np.linalg.norm(leg1['center'][:2] - leg2['center'][:2])

                is_long  = abs(dist - tray_config.SPACING_LONG)  < tray_config.SPACING_TOL
                is_short = abs(dist - tray_config.SPACING_SHORT) < tray_config.SPACING_TOL

                if is_long or is_short:
                    pairs.append({
                        "type": "LONG" if is_long else "SHORT",
                        "legs": [leg1, leg2],
                        "indices": {i, j},
                        "dist": dist
                    })
        return pairs

    def get_flattened_layer(self, pcd, z_min, z_max):
        """Crop to Z slice and flatten all points to Z=0."""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-100, -100, z_min]),
            max_bound=np.array([ 100,  100, z_max])
        )
        pts = np.asarray(pcd.crop(bbox).points)
        if len(pts) > 0:
            pts[:, 2] = 0
        pcd_flat = o3d.geometry.PointCloud()
        if len(pts) > 0:
            pcd_flat.points = o3d.utility.Vector3dVector(pts)
        return pcd_flat
