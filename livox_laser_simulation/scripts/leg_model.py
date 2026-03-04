"""
Leg detection model fitting using Open3D point cloud processing
Creates a 3D template of the C-shaped tray legs for matching against sensor data
"""
import open3d as o3d
import numpy as np
import math

def create_tray_face_template(side="SHORT"):
    """
    Creates an edge-centered 2-leg template for either the 'short' or 'long' side.
    The physical C-shape of the leg is preserved: The X-axis walls are always solid, 
    and the empty face always points inward along the Y-axis.
    """
    # ---------------------------------------------------------
    # 1. MASTER PARAMETERS 
    # ---------------------------------------------------------
    z_low, z_high = 0.05, 0.15
    w_low, d_low = 0.0840, 0.1414
    w_high, d_high = 0.1015, 0.1502

    x_offset_low, x_offset_high = 2.1717, 2.1717  
    y_offset_low, y_offset_high = 0.7692, 0.7736  
    # ---------------------------------------------------------

    # 2. Set coordinates based on which side the robot is facing
    if side == "SHORT":
        # Edge-centered at X=0 (Legs span across the Y-axis)
        legs_low = [[0.0, -y_offset_low], [0.0, y_offset_low]]
        legs_high = [[0.0, -y_offset_high], [0.0, y_offset_high]]
    elif side == "LONG":
        # Edge-centered at Y=0 (Legs span across the X-axis)
        legs_low = [[-x_offset_low, 0.0], [x_offset_low, 0.0]]
        legs_high = [[-x_offset_high, 0.0], [x_offset_high, 0.0]]
    else:
        raise ValueError("Argument 'side' must be 'SHORT' or 'LONG'.")

    all_points = []

    # 3. Generate the 3D points
    for i in range(2):
        cx_low, cy_low = legs_low[i]
        cx_high, cy_high = legs_high[i]
        
        for z in np.linspace(z_low, z_high, 30):
            ratio = (z - z_low) / (z_high - z_low)
            
            cx = cx_low + ratio * (cx_high - cx_low)
            cy = cy_low + ratio * (cy_high - cy_low)
            w = w_low + ratio * (w_high - w_low)
            d = d_low + ratio * (d_high - d_low)
            
            x_min, x_max = cx - w/2, cx + w/2
            y_min, y_max = cy - d/2, cy + d/2
            
            # ---------------------------------------------------------
            # THE TRUE C-SHAPE LOGIC
            # ---------------------------------------------------------
            
            # RULE 1: X-axis walls (Depth) are ALWAYS solid.
            # This forms the solid "inner" and "outer" faces when looking at the long side.
            for y in np.linspace(y_min, y_max, 10):
                all_points.append([x_min, y, z]) # Left X wall
                all_points.append([x_max, y, z]) # Right X wall
                
            # RULE 2: Y-axis walls (Width) have one empty side.
            for x in np.linspace(x_min, x_max, 10):
                if side == "SHORT":
                    # On the short side, the empty faces point toward each other (inward).
                    if cy_low < 0:
                        all_points.append([x, y_min, z]) # Leg 1: Outer -Y wall (Empty +Y)
                    else:
                        all_points.append([x, y_max, z]) # Leg 2: Outer +Y wall (Empty -Y)
                        
                elif side == "LONG":
                    # On the long side, the legs sit at Y=0.
                    # The center of the tray is located deeper in the +Y direction.
                    # Therefore, BOTH legs have their empty side facing +Y (inward to tray).
                    # We ONLY draw the outer -Y wall for both legs.
                    all_points.append([x, y_min, z]) 

    # 4. Build the Open3D PointCloud
    template_pcd = o3d.geometry.PointCloud()
    template_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
    template_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    template_pcd.orient_normals_consistent_tangent_plane(k=15)
    
    return template_pcd

def fit_live_lidar_to_tray(leg1, leg2, side="SHORT", is_under_tray=False, sensor_pos=None):
    """
    Fits two pre-clustered leg point clouds to the Tray Template and extracts the final Pose.

    Args:
        leg1: dict with 'center' (np.array) and 'pcd' (o3d.PointCloud)
        leg2: dict with 'center' (np.array) and 'pcd' (o3d.PointCloud)
        side: Either "SHORT" or "LONG" to select which template to use
        is_under_tray: If True, flips the long-side template 180 deg
        sensor_pos: LiDAR position in the same frame as the point clouds (e.g. odom).
                    If None, defaults to [0, 0, 0.25] (only correct near odom origin).

    Returns:
        final_x: Fitted tray center X position
        final_y: Fitted tray center Y position
        final_yaw: Fitted tray orientation (radians)
        final_transform: 4x4 transformation matrix
    """
    # 1. Create the 3D template (Z=0.05–0.15).
    template_pcd = create_tray_face_template(side=side)

    # 2. Determine left / right from the robot's perspective
    #    Robot looks in +X, so left = larger Y, right = smaller Y
    center_1 = leg1['center'][:2]
    center_2 = leg2['center'][:2]

    if center_1[1] > center_2[1]:
        left_center, right_center = center_1, center_2
    else:
        left_center, right_center = center_2, center_1

    print(f"Left  leg center: X={left_center[0]:.4f}, Y={left_center[1]:.4f}")
    print(f"Right leg center: X={right_center[0]:.4f}, Y={right_center[1]:.4f}")

    # 3. Build the initial guess from the two leg centers
    #    Pass right first, left second so the line_angle points from -Y to +Y
    guess_matrix, guess_yaw = calculate_initial_guess(
        right_center, left_center, side, is_under_tray)

    # 4. Filter template to only truly visible points (two-stage).
    #    Stage A: Backface culling — remove walls whose normals face away
    #             from the sensor (back wall, opposite side wall).
    #    Stage B: HPR — from the remaining front-facing points, remove
    #             parts occluded by other walls (e.g. side wall hidden
    #             behind front face at oblique angles).
    #
    #    SKIP culling when the sensor is between the two legs.
    #    In that case the C-shape is viewed from inside and the normal-based
    #    culling removes the wrong faces.  We detect this by checking the
    #    angle subtended by the two legs as seen from the sensor — if the
    #    angle exceeds 90° the sensor must be between them.
    if sensor_pos is None:
        sensor_pos = np.array([0.0, 0.0, 0.25])
    else:
        sensor_pos = np.asarray(sensor_pos, dtype=float)

    # Check if sensor is between the two legs
    sensor_2d = sensor_pos[:2]
    vec_to_left = left_center - sensor_2d
    vec_to_right = right_center - sensor_2d
    norm_left = np.linalg.norm(vec_to_left)
    norm_right = np.linalg.norm(vec_to_right)
    if norm_left > 1e-6 and norm_right > 1e-6:
        cos_angle = np.dot(vec_to_left, vec_to_right) / (norm_left * norm_right)
    else:
        cos_angle = 1.0  # sensor on top of a leg, treat as outside
    sensor_between_legs = cos_angle < -0.64  # angle > ~130°

    template_world = o3d.geometry.PointCloud(template_pcd)
    template_world.transform(guess_matrix)
    world_pts = np.asarray(template_world.points)
    world_normals = np.asarray(template_world.normals)
    pre_count = len(world_pts)

    if sensor_between_legs:
        # Sensor is between the legs — ICP is unreliable from this viewpoint.
        # Return None to signal the caller to use the last good detection.
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        print(f"[ICP DEBUG] Sensor between legs (angle={angle_deg:.0f}°), skipping ICP")
        return None, None, None, None, -1.0
    else:
        # Stage A: Backface culling
        view_dirs = sensor_pos - world_pts
        view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs = view_dirs / np.clip(view_norms, 1e-8, None)
        dots = np.sum(world_normals * view_dirs, axis=1)
        front_mask = dots > 0.3
        front_indices = np.where(front_mask)[0]
        template_pcd = template_pcd.select_by_index(front_indices.tolist())
        print(f"[ICP DEBUG] After backface cull: {len(front_indices)} / {pre_count} pts")

        # Stage B: HPR on the front-facing subset
        try:
            culled_world = template_world.select_by_index(front_indices.tolist())
            _, hpr_indices = culled_world.hidden_point_removal(sensor_pos.tolist(), 100)
            template_pcd = template_pcd.select_by_index(hpr_indices)
            print(f"[ICP DEBUG] After HPR: {len(hpr_indices)} / {len(front_indices)} pts")
        except RuntimeError:
            print("[ICP DEBUG] HPR failed (too few points), using backface-culled set")

    # 5. Combine both leg pcds into one cloud for ICP (this is the "scene" / target)
    scene_pcd = leg1['pcd'] + leg2['pcd']
    scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)

    # 5b. Scene backface cull — per-leg hemisphere filter.
    #     Accumulated frames may contain points from both front and back faces
    #     of a leg as the robot drives through. For each leg, remove points on
    #     the far side relative to the current sensor position.
    #     Uses geometry (not normals) so it works on sparse lidar data.
    pre_scene = len(np.asarray(scene_pcd.points))
    scene_pts = np.asarray(scene_pcd.points)
    keep_mask = np.ones(len(scene_pts), dtype=bool)
    leg_radius = 0.15

    for leg in [leg1, leg2]:
        lc = leg['center'][:2]
        sensor_dir = sensor_pos[:2] - lc
        sensor_dir_norm = np.linalg.norm(sensor_dir)
        if sensor_dir_norm < 1e-6:
            continue
        sensor_dir = sensor_dir / sensor_dir_norm

        pt_dirs = scene_pts[:, :2] - lc
        pt_dists = np.linalg.norm(pt_dirs, axis=1)
        near_mask = pt_dists < leg_radius

        pt_dir_norms = np.linalg.norm(pt_dirs, axis=1, keepdims=True)
        pt_dirs_n = pt_dirs / np.clip(pt_dir_norms, 1e-8, None)
        cos_angles = np.sum(pt_dirs_n * sensor_dir, axis=1)

        # cos < 0 means the point is on the opposite side of the leg from the sensor
        far_side = near_mask & (cos_angles < 0.0)
        keep_mask &= ~far_side

    scene_pcd = scene_pcd.select_by_index(np.where(keep_mask)[0].tolist())
    print(f"[ICP DEBUG] Scene backface cull: {pre_scene} -> {len(scene_pcd.points)} pts")

    # ---- DEBUG ----
    t_pts = np.asarray(template_pcd.points)
    s_pts = np.asarray(scene_pcd.points)
    print(f"[ICP DEBUG] Template: {len(t_pts)} pts, Scene: {len(s_pts)} pts")
    test_pcd = o3d.geometry.PointCloud(template_pcd)
    test_pcd.transform(guess_matrix)
    dists = np.asarray(test_pcd.compute_point_cloud_distance(scene_pcd))
    print(f"[ICP DEBUG] Template→Scene distances after guess: "
          f"min={dists.min():.4f}, median={np.median(dists):.4f}, "
          f"max={dists.max():.4f}")
    # ---- END DEBUG ----

    # 6. Run Point-to-Point ICP with the HPR-filtered template.
    #    Only visible faces remain, so there is no C-shape bias.
    threshold = 0.12
    reg_p2l = o3d.pipelines.registration.registration_icp(
        template_pcd,
        scene_pcd,
        threshold,
        guess_matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    # 7. Extract the Final Pose (transformation maps template → world)
    #    ICP runs 6-DOF but we only want XY + yaw.
    #    Rebuild a clean 2D rigid transform to discard any roll/pitch/Z drift.
    raw_transform = reg_p2l.transformation
    final_x = raw_transform[0, 3]
    final_y = raw_transform[1, 3]
    final_yaw = math.atan2(raw_transform[1, 0], raw_transform[0, 0])

    cos_y = math.cos(final_yaw)
    sin_y = math.sin(final_yaw)
    final_transform = np.eye(4)
    final_transform[0, 0] = cos_y
    final_transform[0, 1] = -sin_y
    final_transform[1, 0] = sin_y
    final_transform[1, 1] = cos_y
    final_transform[0, 3] = final_x
    final_transform[1, 3] = final_y

    print(f"ICP Fitness Score: {reg_p2l.fitness:.4f} (Higher is better, 1.0 is perfect)")
    print(f"Final Tray Center: X={final_x:.4f}, Y={final_y:.4f}, Yaw={math.degrees(final_yaw):.2f} deg")

    return final_x, final_y, final_yaw, final_transform, reg_p2l.fitness


def create_leg_markers(final_transform, side, header, ns="fitted_legs", marker_id_start=0):
    """
    Creates RViz markers showing the fitted leg template at the detected pose.

    Args:
        final_transform: 4x4 transformation matrix from ICP
        side: "SHORT" or "LONG"
        header: std_msgs/Header for the markers
        ns: Marker namespace
        marker_id_start: Starting marker ID to avoid conflicts

    Returns:
        (markers, next_marker_id): list of Marker msgs and next available ID
    """
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    import rospy

    markers = []
    marker_id = marker_id_start

    # 1. Create template and transform it to the detected pose
    template_pcd = create_tray_face_template(side=side)
    template_pcd.transform(final_transform)
    pts = np.asarray(template_pcd.points)

    # 2. POINTS marker showing the fitted C-shape template
    points_marker = Marker()
    points_marker.header = header
    points_marker.ns = ns
    points_marker.id = marker_id
    marker_id += 1
    points_marker.type = Marker.POINTS
    points_marker.action = Marker.ADD
    points_marker.pose.orientation.w = 1.0
    points_marker.scale.x = 0.008
    points_marker.scale.y = 0.008
    # Cyan for fitted template
    points_marker.color.r = 0.0
    points_marker.color.g = 1.0
    points_marker.color.b = 1.0
    points_marker.color.a = 0.9
    points_marker.lifetime = rospy.Duration(0)

    for p in pts:
        point = Point()
        point.x = p[0]
        point.y = p[1]
        point.z = p[2]
        points_marker.points.append(point)

    markers.append(points_marker)

    # 3. Text label with pose info
    final_x = final_transform[0, 3]
    final_y = final_transform[1, 3]
    final_yaw = math.atan2(final_transform[1, 0], final_transform[0, 0])

    text_marker = Marker()
    text_marker.header = header
    text_marker.ns = ns + "_labels"
    text_marker.id = marker_id
    marker_id += 1
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = final_x
    text_marker.pose.position.y = final_y
    text_marker.pose.position.z = 0.25
    text_marker.text = f"ICP {side}\nX:{final_x:.3f} Y:{final_y:.3f}\nYaw:{math.degrees(final_yaw):.1f} deg"
    text_marker.scale.z = 0.08
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 1.0
    text_marker.color.a = 1.0
    text_marker.lifetime = rospy.Duration(0)

    markers.append(text_marker)

    return markers, marker_id


import numpy as np
import math

def calculate_initial_guess(cluster_1_center, cluster_2_center, side="SHORT", is_under_tray=False):
    """
    Calculates the 4x4 initial guess matrix for ICP based on two LiDAR clusters.
    Adjusts the Yaw calculation depending on which face (short or long) is detected.
    """
    c1_x, c1_y = cluster_1_center
    c2_x, c2_y = cluster_2_center

    # 1. Calculate the Midpoint (The X, Y translation guess)
    mid_x = (c1_x + c2_x) / 2.0
    mid_y = (c1_y + c2_y) / 2.0

    # 2. Calculate the angle of the line connecting the two LiDAR clusters
    line_angle = math.atan2(c2_y - c1_y, c2_x - c1_x)
    
    # 3. Apply the correct rotation offset based on the template's native state
    if side == "SHORT":
        # Native SHORT template legs are on the Y-axis (pi/2)
        guess_yaw = line_angle - (math.pi / 2.0)
    elif side == "LONG":
        # Native template legs are on the X-axis (0 radians)
        guess_yaw = line_angle
        # If we are under the tray looking at the long side, the hollow C-shape 
        # is facing us. We MUST spin the model 180 degrees so the template's 
        # hollow side faces the LiDAR points.
        if is_under_tray:
            guess_yaw += math.pi
    else:
        raise ValueError("Argument 'side' must be 'SHORT' or 'LONG'.")

    # 4. Build the 4x4 Transformation Matrix
    initial_guess = np.eye(4)
    initial_guess[0, 3] = mid_x
    initial_guess[1, 3] = mid_y
    
    # Apply the Yaw rotation
    cos_y = math.cos(guess_yaw)
    sin_y = math.sin(guess_yaw)
    
    initial_guess[0, 0] = cos_y
    initial_guess[0, 1] = -sin_y
    initial_guess[1, 0] = sin_y
    initial_guess[1, 1] = cos_y

    return initial_guess, guess_yaw