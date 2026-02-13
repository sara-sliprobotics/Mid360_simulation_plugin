import csv

input_file = 'mid360-real-centr.csv'  # Your uploaded file name
output_file = 'mid180.csv'

# We want to keep the front 180 degrees (-90 to +90)
min_deg = -90.0
max_deg = 90.0

count = 0
total = 0

print(f"Reading {input_file}...")

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    # Write the header first
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        total += 1
        try:
            # Column 1 is 'Azimuth/deg'
            azimuth = float(row[1])
            
            # Normalize angle to -180...180 range
            # (Just in case the file has 0...360 format)
            if azimuth > 180: azimuth -= 360
            if azimuth < -180: azimuth += 360
            
            # FILTER: Keep only front 180 degrees
            if min_deg <= azimuth <= max_deg:
                writer.writerow(row)
                count += 1
        except ValueError:
            continue

print(f"Done.")
print(f"Original Lines: {total}")
print(f"New Lines:      {count}")
print("-" * 30)
print(f"Your NEW Gazebo Setting should be approx:")
print(f"<samples>{int(count / 33)}</samples>") 
# Dividing by ~33 because the original file was ~33 frames long (800k / 24k)