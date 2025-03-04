import numpy as np
import matplotlib.pyplot as plt

# --- PARAMETERS ---
map_size = 10  # Map size in meters
resolution = 0.05  # Grid resolution (meters per cell)
robot_pose = (0, 0, 0)  # Robot initial position (x, y, theta)
max_range = 10  # Max range of the laser

# --- Parse the laser scan data from the file ---
def parse_laser_data_from_file(filename):
    laser_scans = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.startswith("ROBOTLASER"):
            values = line.split()
            angles = np.linspace(-2.356194, 2.356194, 541)  # Angle range from data
            ranges = np.array([float(v) for v in values[8:8+541]])  # Extract ranges
            laser_scans.append((angles, ranges))
    
    return laser_scans

# --- Read laser scan data from file ---
file_path = "SLAM_MAP\LaserData.txt"  # Replace with your actual file path

laser_scans = parse_laser_data_from_file(file_path)

# Convert all laser scans to Cartesian coordinates
all_x_points = []
all_y_points = []
for angles, ranges in laser_scans:
    valid = (ranges > 0) & (ranges < max_range)  # Filter valid ranges
    x_points = ranges[valid] * np.cos(angles[valid]) + robot_pose[0]
    y_points = ranges[valid] * np.sin(angles[valid]) + robot_pose[1]
    
    all_x_points.extend(x_points)
    all_y_points.extend(y_points)

# Convert to occupancy grid
grid_size = int(map_size / resolution)
occupancy_grid = np.zeros((grid_size, grid_size))

# Transform coordinates to grid indices
for x, y in zip(all_x_points, all_y_points):
    grid_x = int((x + map_size / 2) / resolution)
    grid_y = int((y + map_size / 2) / resolution)
    
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        occupancy_grid[grid_y, grid_x] = 1  # Mark occupied

# --- PLOT OCCUPANCY GRID ---
plt.imshow(occupancy_grid, cmap="gray_r", origin="lower")
plt.xlabel("X (cells)")
plt.ylabel("Y (cells)")
plt.title("Occupancy Grid Map")
plt.show()
