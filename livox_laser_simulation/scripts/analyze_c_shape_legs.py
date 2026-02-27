import trimesh
import numpy as np
import os

def analyze_c_shape_legs(stl_filename='tray.stl', z_low=0.05, z_high=0.15):
    if not os.path.exists(stl_filename):
        print(f"Error: Could not find '{stl_filename}'.")
        return

    print(f"--- Analyzing C-Shape Legs in: {stl_filename} ---")
    mesh = trimesh.load(stl_filename)

    def get_slice_data(z_height):
        # 1. Cut the mesh at the specific Z height
        section = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
        if section is None:
            return []
        
        # 2. Convert to 2D paths
        slice_2d, _ = section.to_2D()
        
        # 3. For C-shaped legs, we have 12 polygons (3 per leg)
        # Group them by proximity to find the 4 legs
        all_polys = []
        for poly in slice_2d.polygons_full:
            minx, miny, maxx, maxy = poly.bounds
            center_x = (minx + maxx) / 2.0
            center_y = (miny + maxy) / 2.0
            all_polys.append({
                'bounds': poly.bounds,
                'center': np.array([center_x, center_y]),
                'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy
            })
        
        # Group polygons into 4 legs based on proximity (within ~0.2m)
        legs = []
        used = [False] * len(all_polys)
        
        for i, poly in enumerate(all_polys):
            if used[i]:
                continue
            
            # Start a new leg group
            leg_polys = [poly]
            used[i] = True
            
            # Find all nearby polygons (same leg)
            for j, other in enumerate(all_polys):
                if used[j]:
                    continue
                dist = np.linalg.norm(poly['center'] - other['center'])
                if dist < 0.2:  # 20cm threshold for same leg
                    leg_polys.append(other)
                    used[j] = True
            
            # Compute overall bounding box for this leg
            minx = min(p['minx'] for p in leg_polys)
            miny = min(p['miny'] for p in leg_polys)
            maxx = max(p['maxx'] for p in leg_polys)
            maxy = max(p['maxy'] for p in leg_polys)
            
            width = maxx - minx
            depth = maxy - miny
            center_x = (minx + maxx) / 2.0
            center_y = (miny + maxy) / 2.0
            
            legs.append({
                'center': np.array([center_x, center_y]),
                'width': width,
                'depth': depth
            })
        
        # Sort left-to-right, then bottom-to-top to keep leg ordering consistent
        legs.sort(key=lambda l: (l['center'][0], l['center'][1]))
        return legs

    # Run the slice analysis
    legs_low = get_slice_data(z_low)
    legs_high = get_slice_data(z_high)

    print(f"Detected {len(legs_low)} legs at Z = {z_low}m")
    print(f"Detected {len(legs_high)} legs at Z = {z_high}m\n")

    if len(legs_low) != 4 or len(legs_high) != 4:
        print("Warning: Did not detect exactly 4 legs at both heights. Check your Z-heights or STL units.")
        return

    # Match and Print Parameters
    for i in range(4):
        low = legs_low[i]
        
        # Find the matching top slice for this specific bottom slice
        distances = [np.linalg.norm(low['center'] - high['center']) for high in legs_high]
        high = legs_high[np.argmin(distances)]
        
        print(f"========== LEG {i+1} ==========")
        print(f"[Z = {z_low}m] Center: (X: {low['center'][0]:.4f}, Y: {low['center'][1]:.4f})")
        print(f"            Size:   Width = {low['width']:.4f}, Depth = {low['depth']:.4f}")
        print(f"[Z = {z_high}m] Center: (X: {high['center'][0]:.4f}, Y: {high['center'][1]:.4f})")
        print(f"            Size:   Width = {high['width']:.4f}, Depth = {high['depth']:.4f}")
        
        # Determine if it's square or rectangular based on a 2mm tolerance
        shape_type = "Square" if abs(low['width'] - low['depth']) < 0.002 else "Rectangular"
        print(f"            Shape:  {shape_type}")
        print("==============================\n")

if __name__ == "__main__":
    stl_path = '/home/slip/rapid-docker/src/external_packages/Mid360_simulation_plugin/livox_laser_simulation/models/tray/meshes/tray.stl'
    # Analyze at the specified z-levels
    analyze_c_shape_legs(stl_path, z_low=0.05, z_high=0.15)
