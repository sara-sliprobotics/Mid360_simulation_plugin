# Object Detection Methods — Mid360 Simulation Plugin

This document describes how **walls**, **trucks (trailers)**, **trays**, and **persons** are detected from the Livox Mid-360 3D LiDAR point cloud. All detection runs inside the `LivoxObjectDetector` node ([object_detection.py](livox_laser_simulation/scripts/object_detection.py)).

---

## System Overview

```
/livox/lidar (PointCloud2, base_link frame)
       │
       ├──► FrameAccumulator ──► rolling buffer of N frames in odom frame
       │                              (used by Tray & Leg detectors)
       │
       ▼
LivoxObjectDetector.pointcloud_callback()
       │
       ├─ 1. WallDetector.detect(points)        → walls[]
       ├─ 2. TruckDetector.detect(walls)         → trucks[]
       ├─ 3. PersonDetector.detect(pcd)           → persons[]
       ├─ 4. LegFirstTrayDetector.detect(walls)   → trays[]
       └─ 5. LegDetector.detect(box per tray)     → legs[] (ICP-fitted poses)
              │
              ▼
       Publish: /livox/detected_objects (MarkerArray)
       Publish: /livox/detections       (JSON String)
       Publish: /livox/accumulated_cloud (PointCloud2, odom)
```

**Input topic:** `/livox/lidar` (PointCloud2)
**Frame accumulator:** Transforms each frame into `odom` via TF and keeps a rolling buffer (default 5 frames) so tray/leg detection operates on a denser, spatially-consistent cloud.

---

## 1. Wall Detection

**File:** [wall_detector.py](livox_laser_simulation/scripts/wall_detector.py)
**Method:** Iterative RANSAC plane segmentation with connectivity filtering

### Summary

Walls are detected by repeatedly fitting planes (RANSAC) to the point cloud and keeping only those that are vertical and spatially connected. Each iteration removes the found plane so the next-largest plane can be discovered.

### Step-by-Step

1. **Voxel downsample** the input cloud (5 cm grid) to reduce computation.

2. **RANSAC loop** — repeat until the remaining cloud has fewer than `min_points` (200):

   a. **Fit the largest plane** using RANSAC (3 sample points, 1000 iterations, 10 cm inlier threshold).

   b. **Stop** if the plane has fewer than 200 inliers (only noise remains).

   c. **Check orientation** — extract the normal vector `[a, b, c]` from the plane equation `ax + by + cz + d = 0`. A wall is vertical, so its normal points sideways: **`|c| < 0.2`**. If `|c| >= 0.2` it is a floor/ceiling and is discarded.

   d. **Connectivity filter (DBSCAN)** — cluster the inlier points spatially (`eps=0.5 m`, `min_points=5`) and keep only the **largest connected cluster**. This prevents merging a wall with a disconnected object (e.g. a tray on the floor) that happens to lie on the same geometric plane.

   e. **Width filter** — compute the axis-aligned bounding box of the cluster. Reject if `max(width_x, width_y) < 2.0 m` (too narrow to be a real wall).

   f. **Store** the wall: plane equation, points, normal, and point count.

   g. **Remove all inliers** (wall or floor) from the cloud and repeat.

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `voxel_size` | 0.05 m | Downsample grid size |
| `distance_threshold` | 0.1 m | RANSAC inlier tolerance |
| `min_points` | 200 | Min inliers for a valid plane |
| `vertical_threshold` | 0.2 | Max `|c|` for a vertical plane |
| `ransac_n` | 3 | Points per RANSAC sample |
| `num_iterations` | 1000 | RANSAC attempts |
| `min_wall_width` | 2.0 m | Min bounding-box extent |
| `connectivity_eps` | 0.5 m | DBSCAN cluster radius |

### Output

List of wall dictionaries: `{plane_model, points, num_points, normal}`

---

## 2. Truck (Trailer) Detection

**File:** [truck_detector.py](livox_laser_simulation/scripts/truck_detector.py)
**Method:** Parallel wall-pair matching with width constraint

### Summary

A truck/trailer is identified when two detected walls are parallel, both long enough, and separated by a distance matching the expected trailer width (~2.6 m).

### Step-by-Step

1. **Iterate all wall pairs** `(i, j)`, skipping walls already matched.

2. **Parallelism check** — normalize both normal vectors and compute the angle between them:
   ```
   angle = arccos(|dot(n1, n2)|)
   ```
   Accept if `angle < 0.1 rad` (~5.7 deg).

3. **Length check** — for each wall, compute `max(extent_x, extent_y)` from the bounding box. Both walls must be >= 3.0 m.

4. **Distance check** — compute perpendicular distance between the two planes:
   ```
   distance = |d1/||n1|| - d2/||n2|||
   ```
   Accept if `|distance - 2.6| <= 0.3` (i.e. 2.3 m to 2.9 m).

5. **Determine front/back walls** — the wall whose centroid is closer to the sensor origin `(0,0,0)` is the front wall.

6. **Compute truck geometry:**
   - **Front edge:** closest point on the front wall to the sensor.
   - **Wall direction:** perpendicular to the wall normal, `[-n_y, n_x, 0]` (normalized).
   - **Depth direction:** unit vector from front wall centroid to back wall centroid.
   - **Truck center:** front edge + half truck length along the wall + half width in the depth direction. Z is set to 1.25 m (mid-height of the 2.5 m trailer).

7. **Mark walls as used** so they are not reused in another truck.

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `truck_width` | 2.6 m | Expected wall-to-wall distance |
| `width_tolerance` | 0.3 m | Acceptable deviation from expected width |
| `parallel_angle_threshold` | 0.1 rad | Max angle to consider walls parallel |
| `min_wall_length` | 3.0 m | Min extent for a truck wall |
| `truck_length` | 16.14 m | Hardcoded from model (used for center calculation) |
| `truck_height` | 2.5 m | Hardcoded from model |

### Output

List of truck dictionaries: `{wall_pair, center, front_edge, width, length, height, orientation, direction, walls}`

---

## 3. Tray Detection

**Files:** [tray_detector.py](livox_laser_simulation/scripts/tray_detector.py), [leg_detector.py](livox_laser_simulation/scripts/leg_detector.py), [leg_model.py](livox_laser_simulation/scripts/leg_model.py), [tray_config.py](livox_laser_simulation/scripts/tray_config.py)
**Method:** "Leg-First" geometric pattern matching + edge verification + ICP pose refinement

### Summary

Trays are detected by their **legs** — small objects near the ground. The algorithm finds leg candidates by clustering low-height points, then matches them into pairs/corners at known tray-leg spacings. An "edge verification" step confirms that a structural bar connects the legs above them. Finally, an ICP fit against a C-shaped leg template produces a precise pose.

### Tray Geometry (from STL)

| Dimension | Value |
|---|---|
| Full length (X) | 5.182 m |
| Full width (Y) | 2.473 m |
| Long-side leg spacing | 4.343 m |
| Short-side leg spacing | 1.558 m |
| Leg height zone | 0.05 m – 0.15 m |
| Edge (bar) height zone | 0.28 m – 0.45 m |
| Max leg bounding box | 0.25 m |
| Min leg bounding box | 0.02 m |

### Step-by-Step

#### Phase 1: Point Cloud Preparation

1. **Use accumulated cloud** — the `FrameAccumulator` provides a merged multi-frame cloud in the `odom` frame (denser than a single scan).

2. **Remove wall points** — for each detected wall, compute point-to-plane distance for every point in the cloud. Remove points within 0.15 m of any wall plane. This prevents wall points from being mistaken for legs.

#### Phase 2: Find Leg Candidates (LegDetector)

3. **Z-slice** the cloud to the leg height zone (0.05 m – 0.15 m) and **flatten** all points to Z=0 (treat as 2D problem).

4. **DBSCAN clustering** (`eps=0.05 m`, `min_points=5`) on the flattened points.

5. **Size filter** each cluster — keep only those with bounding box extent between 0.02 m and 0.25 m (small objects = legs). Reject clusters with fewer than 15 points.

#### Phase 3: Pattern Matching (Pairs and Corners)

6. **Find pairs** — for every combination of two leg candidates, check if their XY distance matches either:
   - **Short-side spacing:** 1.558 m +/- 0.20 m
   - **Long-side spacing:** 4.343 m +/- 0.20 m

7. **Find corners (3-leg L-shapes)** — if two pairs share exactly one leg, three legs form an L-shape (two perpendicular sides of the tray). For each potential corner:

   a. **Arrow Test (convex vs concave)** — compute the corner's bisector direction `-(V1 + V2)` and check if it points toward the robot (dot product with vector-to-robot > 0.2). A tray corner points **at** the robot (convex); a room corner points **away** (concave). Reject concave corners.

   b. **Edge verification** — slice the cloud at 0.28 m – 0.45 m (the tray's structural bar height). Check that a line of at least 20 points connects the corner leg to each end leg (within 0.5 m perpendicular distance). This confirms a physical bar exists above the legs.

   c. **Compute tray center** — estimate the 4th leg position: `fourth = corner + (end1 - corner) + (end2 - corner)`. Center = average of all 4 positions.

8. **Process remaining 2-leg pairs** — pairs not consumed by a corner detection are checked individually:
   - Run the same edge verification.
   - Estimate tray center by shifting the pair midpoint perpendicular to the pair line by half the other leg spacing, away from the robot.
   - Mark used legs to prevent double detection.

#### Phase 4: ICP Pose Refinement (LegDetector + LegModel)

9. For each detected tray, create a **bounding box** around the tray center and run the full `LegDetector.detect()` pipeline within it:

   a. Find leg candidates (with 3D points preserved).

   b. Find the best-matching pair for the requested enter side.

   c. **Build a C-shape template** of the tray legs at the correct spacing using `create_tray_face_template()`. The template models the physical C-shape of each leg (solid X-walls, open inner Y-face).

   d. **Compute initial guess** — midpoint of the two detected leg centroids gives XY; the angle of the line between them gives yaw (adjusted by 90 deg for short side).

   e. **Visibility filtering** on the template:
      - **Backface culling:** remove template points whose normals face away from the sensor (dot product < 0.3).
      - **Hidden Point Removal (HPR):** remove template points occluded by other parts of the template from the sensor's viewpoint.

   f. **Point-to-Point ICP** — register the visibility-filtered template against the combined scene cloud of both legs (threshold=0.12 m, max 50 iterations).

   g. **Extract pose** — from the 4x4 result matrix, take `(X, Y, yaw)` and rebuild a clean 2D rigid transform (discard roll/pitch/Z drift).

### Output

- **Trays** (`/livox/detections` JSON): center position, detection type (`TRAY_CORNER`, `TRAY_SIDE_SHORT`, `TRAY_SIDE_LONG`), number of legs.
- **Legs** (`/livox/detections` JSON): ICP-fitted center (x, y) and yaw for each confirmed leg pair.

---

## 4. Person Detection

**File:** [person_detector.py](livox_laser_simulation/scripts/person_detector.py)
**Method:** Dimensional constraint filtering + DBSCAN clustering + motion tracking

### Summary

Persons are detected by clustering the full-height point cloud (above the floor) and checking each cluster against human body dimension rules: tall, narrow, and taller-than-wide. An optional motion filter tracks detections over time and only reports persons who are moving.

### Step-by-Step

1. **Remove floor** — crop the point cloud to `0.1 m <= Z <= 2.5 m`. This eliminates ground points and anything above ceiling height.

2. **DBSCAN clustering** (`eps=0.25 m`, `min_points=50`) to group nearby points into distinct objects.

3. **Human dimension check** on each cluster (3 rules applied in order):

   | Rule | Check | Threshold | Rejects |
   |---|---|---|---|
   | 1. Height | `height >= min_human_height` | 1.40 m | Robots, trays, furniture |
   | 2. Width | `max_width <= max_human_width` | 0.80 m | Walls, large objects |
   | 3. Aspect ratio | `height > max_width` | — | Squat/flat objects |

   If all 3 rules pass, the cluster is classified as a person.

4. **Motion filtering** (when `require_motion=True`):

   a. **Match to existing tracks** — for each raw detection, find the closest tracked person within 1.0 m (XY distance). If matched, update the track.

   b. **Movement check:**
      - If the person moved more than `motion_threshold` (0.1 m) since last seen: mark as **moving**, include in output.
      - If stationary but within the `stationary_timeout` grace period (2.0 s): still include (recently stopped).
      - If stationary longer than the timeout: **filter out** (static object, not a person).

   c. **New detections** are assumed moving and added to tracking.

   d. **Track cleanup** — remove tracks not updated for 5.0 seconds.

### Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `min_human_height` | 1.40 m | Min cluster height to be human |
| `max_human_width` | 0.80 m | Max cluster width to be human |
| `min_points` | 50 | Min points in cluster |
| `eps` | 0.25 m | DBSCAN clustering distance |
| `max_height` | 2.5 m | Ceiling cutoff |
| `require_motion` | True | Enable motion-based filtering |
| `motion_threshold` | 0.1 m | Min movement to be "moving" |
| `stationary_timeout` | 2.0 s | Grace period for stationary persons |

### Output

List of person dictionaries: `{center, height, width, points, cluster, is_moving, min_bound, max_bound}`

---

## 5. Frame Accumulator (Supporting Component)

**File:** [frame_accumulator.py](livox_laser_simulation/scripts/frame_accumulator.py)

The `FrameAccumulator` subscribes to `/livox/lidar` and maintains a **rolling buffer** (default 5 frames). Each incoming frame is transformed from the sensor frame into the `odom` frame using TF2, then stored. When a detector calls `get_accumulated_cloud()`, all buffered frames are merged into a single dense cloud. This gives the tray/leg detectors much more data to work with than a single scan.

---

## Published Topics

| Topic | Type | Frame | Content |
|---|---|---|---|
| `/livox/detected_objects` | MarkerArray | mixed | RViz visualization (walls=red cubes, trucks=blue boxes, persons=red cylinders, trays=green boxes, legs=cyan points) |
| `/livox/detections` | String (JSON) | — | Structured results: walls, trucks, persons, trays, legs with positions and metadata |
| `/livox/accumulated_cloud` | PointCloud2 | odom | Debug cloud showing accumulated frames |

## Subscribed Topics

| Topic | Type | Purpose |
|---|---|---|
| `/livox/lidar` | PointCloud2 | Raw 3D LiDAR input |
