import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils 
import time
import scipy.ndimage as ndimg
import copy


def convert_to_grid_map():
    # Load the image
    image_path = 'SLAM_MAP/artifitial_SLAM_MAP_2.png'  # Replace with your image file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        exit()

    # Define the pixel values for free, occupied, and unknown areas
    FREE = 255      # White in grayscale
    OCCUPIED = 0    # Black in grayscale

    # Create the occupancy grid
    occupancy_grid = np.zeros_like(image, dtype=float)


    # Assign values based on pixel intensity
    occupancy_grid[image == FREE] = 1       # Free space
    occupancy_grid[image == OCCUPIED] = 0   # Occupied space (obstacles)
    occupancy_grid[(image < 200) & (image > 100)] = 0.5   # Unknown space


    # Display the occupancy grid
    plt.imshow(occupancy_grid, cmap='gray', vmin=0, vmax=1)
    plt.title('Occupancy Grid Map')
    plt.colorbar(label='Occupancy (1=Free, 0=Occupied, 0.5=Unknown)')
    plt.show()

    # Save the occupancy grid as a CSV file
    np.savetxt('SLAM_MAP/occupancy_grid.csv', occupancy_grid, delimiter=',', fmt='%.1f')

    # Save the occupancy grid as an image (optional)
    cv2.imwrite('SLAM_MAP/occupancy_grid_visualization.png', (occupancy_grid) * 255)  # Scale to 0-255 for visualization


def load_grid_map(visulize = False):
    # Load the occupancy grid from the CSV file
    occupancy_grid = np.loadtxt('SLAM_MAP/occupancy_grid.csv', delimiter=',')

    if visulize:
        plt.imshow(occupancy_grid, cmap='gray', vmin=0, vmax=1)
        plt.title('Occupancy Grid Map')
        plt.colorbar(label='Occupancy (1=Free, 0=Occupied, 0.5=Unknown)')
        plt.show()

    occupancy_grid[occupancy_grid != 1] = 0
    occupancy_grid[occupancy_grid == 1] = 0.5

    return occupancy_grid



class Camera:
    def __init__(self, position, FOV, max_distance):
        self.position = position
        self.FOV = FOV
        self.max_distance = max_distance

    def get_visible_points(self, grid_map):
        points = utils.bresenham_circle_segment(self.position[0], self.position[1], self.max_distance)
        points = np.array(points)
        yaw = (self.position[3])
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rotated_points = np.dot(points - self.position[:2], rotation_matrix.T) + self.position[:2]
        points = rotated_points

        visible_points = []
        for point in points:
            line = utils.bresenham_line(self.position, point)
            # line = utils.dda_line(self.position, point)

            for i, line_point in enumerate(line):
                if grid_map[line_point[1], line_point[0]] == 0:
                    break
                grid_map[line_point[1], line_point[0]] = 1
                visible_points.append(line_point)

        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        convolved_grid = cv2.filter2D(grid_map, -1, kernel)
        grid_map[(convolved_grid > 5) & (convolved_grid >= 0.35)] = 1

        
        return visible_points
    
    def plot_camera_position(self):
        # Plot the camera position on the map
        plt.scatter(self.position[0], self.position[1], c='red', marker='o', label='Camera Position')
        
        end_x = self.position[0] + 50 * np.cos((self.position[3]))
        end_y = self.position[1] + 50 * np.sin((self.position[3]))

        # Calculate the FOV lines
        left_FOV_x = self.position[0] + 50 * np.cos((self.position[3] - self.FOV / 2))  
        left_FOV_y = self.position[1] + 50 * np.sin((self.position[3] - self.FOV / 2))
        right_FOV_x = self.position[0] + 50 * np.cos((self.position[3] + self.FOV / 2))
        right_FOV_y = self.position[1] + 50 * np.sin((self.position[3] + self.FOV / 2))

        # Plot the FOV lines
        plt.plot([self.position[0], left_FOV_x], [self.position[1], left_FOV_y], c='green', linestyle='--', label='Left FOV')
        plt.plot([self.position[0], right_FOV_x], [self.position[1], right_FOV_y], c='green', linestyle='--', label='Right FOV')
        end_y = self.position[1] + 50 * np.sin((self.position[3]))
        # Plot the facing direction
        plt.plot([self.position[0], end_x], [self.position[1], end_y], c='blue', label='Facing Direction')

    def move(self, dx, dy, dz):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
    
    def rotate(self, dyaw, dpitch, droll):
        self.position[3] += dyaw
        self.position[4] += dpitch
        self.position[5] += droll
    
    def set_position(self, position):
        self.position = position
    
    def set_FOV(self, FOV):
        self.FOV = FOV

    def set_max_distance(self, max_distance):
        self.max_distance = max_distance

    def scan_circle(self, grid_map):
        overlay = np.deg2rad(15)
        one_rotation = self.FOV - overlay
        rotations = np.ceil(2 * np.pi / one_rotation)
        for i in range(int(rotations)):
            self.get_visible_points(grid_map)
            self.rotate(one_rotation, 0, 0)
        




def main():
    plt.figure()
    grid_map = load_grid_map()

    camera_position = [400, 400, 100, 0, 0, 0] # [x, y, z, yaw, pitch, roll]
    camera_FOV = np.deg2rad(60)  # 60 degrees
    max_distance = 300 # 3 meters

    camera = Camera(camera_position, camera_FOV, max_distance)
    camera.scan_circle(grid_map)    
    
    gridmap_unkown = np.where((grid_map > 0.35) & (grid_map < 0.65), 0, 1)
    gridmap_obstacles = np.where(grid_map < 0.35, 0, 1)

    
    samp = 0.02
    gridmap_obstacles = ndimg.distance_transform_edt(gridmap_obstacles, sampling = [samp, samp])
    gridmap_obstacles = np.where(gridmap_obstacles > 0.5, 1., 0.)

    gridmap_unkown = ndimg.distance_transform_edt(gridmap_unkown, sampling = [samp, samp])
    gridmap_unkown = np.where(gridmap_unkown > 0.5, 1., 0.)


    plt.subplot(2, 2, 1)
    camera.plot_camera_position()  
    # plt.legend()
    plt.imshow(gridmap_unkown, cmap='gray')
    plt.title('unknown')

    plt.subplot(2, 2, 2)
    camera.plot_camera_position()  
    # plt.legend()
    plt.imshow(gridmap_obstacles, cmap='gray')
    plt.title('obstacles')

    
    grid_map_inflated = np.where(gridmap_obstacles == 0, 0, np.where(gridmap_unkown == 0, 0.5, 1))

    plt.subplot(2, 2, 3)
    camera.plot_camera_position()  
    # plt.legend()  
    plt.imshow(grid_map_inflated, cmap='gray')
    plt.title('grid_map_inflated')


    gridmap_copy = grid_map.copy()
    a = 0
    b = 1
    mask = np.array([[b, b, b], [b, a, b], [b, b, b]])


    # plt.imshow(gridmap_copy, cmap='gray', vmin=0, vmax=1)

    data_c = ndimg.convolve(gridmap_copy, mask, mode='constant', cval=0.0)


    
    filtered_data_c = np.where((data_c >= 5.5) & (data_c <= 6), 1, 0)

    
    

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_data_c, cmap='gray')
    plt.title('Filtered Grid Map')
    plt.imshow(filtered_data_c, cmap='gray')


    plt.title('Occupancy Grid Map with Camera Position')
    plt.show()



if __name__ == "__main__":
    main()