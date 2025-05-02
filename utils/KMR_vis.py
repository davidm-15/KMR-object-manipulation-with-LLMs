import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import os

# ==== Configuration ====
# Robot dimensions (mm)
robot_length = 1130
robot_width = 630
robot_height = 700

# Mobile base position and orientation (center on ground)
robot_pos_x = 13804   # mm
robot_pos_y = 15117  # mm
robot_theta_base = -3.1126 # radians (rotation around Z-axis)

# Manipulator base offset relative to mobile base center (X=length, Y=width, Z=height)
manipulator_base_offset_local = np.array([363, -184, robot_height]) # [x, y, z] in robot's frame

# --- IIWA Manipulator ---
# DH Parameters (Standard DH: [a, d, alpha, theta_offset]) - ORDER CORRECTED!
# Input was [theta, d, a, alpha], needs to be [a, d, alpha, theta_offset] for our dh_matrix function
# Original input format:
# dh_params_input = np.array([
#     [0, 360, 0, -np.pi/2],
#     [0, 0, 0, np.pi/2],
#     [0, 420, 0, np.pi/2],
#     [0, 0, 0, -np.pi/2],
#     [0, 400, 0, -np.pi/2],
#     [0, 0, 0, np.pi/2],
#     [0, 152, 0, 0]
# ])
# Corrected order [a, d, alpha, theta_offset]:
dh_params = np.array([
    # a    d    alpha      theta_offset
    [0,   360, -np.pi/2,  0], # Original row 1, swapped columns 0&2, 2&3
    [0,   0,    np.pi/2,  0], # Original row 2, swapped columns 0&2, 2&3
    [0,   420,  np.pi/2,  0], # Original row 3, swapped columns 0&2, 2&3
    [0,   0,   -np.pi/2,  0], # Original row 4, swapped columns 0&2, 2&3
    [0,   400, -np.pi/2,  0], # Original row 5, swapped columns 0&2, 2&3
    [0,   0,    np.pi/2,  0], # Original row 6, swapped columns 0&2, 2&3
    [0,   152,  0,        0]  # Original row 7, swapped columns 0&2, 2&3
])
# Note: Assumed theta_offset is 0 here. If the first column in your source
# actually represented offsets, add them here instead of 0.


# Example Joint Angles (in radians) - MODIFY THESE TO CHANGE THE ARM POSE
joint_angles = np.deg2rad([0, 30, 0, -60, 0, 45, 0])
joint_angles = np.array([-1.218284485, -1.072156, -0.219442, 1.609778,  0.316876, 1.039710, 0.035448])

# Load the hand poses from file
handposes_path = os.path.join('communication', 'HandPoses.json')
try:
    with open(handposes_path, 'r') as f:
        handposes = json.load(f)
except FileNotFoundError:
    print(f"Warning: {handposes_path} not found. Using default pose.")
    handposes = []

# Function to list available poses
def list_poses():
    print("Available poses:")
    for i, pose in enumerate(handposes):
        print(f"[{i}] {pose.get('name', 'unnamed')} (timestamp: {pose.get('timestamp', 'unknown')})")

# Function to set the joint angles from a pose by index
def set_pose_by_index(index):
    global joint_angles, robot_pos_x, robot_pos_y, robot_theta_base
    
    if not handposes or index >= len(handposes):
        print(f"Invalid pose index: {index}. Using default pose.")
        return
    
    pose = handposes[index]
    print(f"Setting to pose: {pose.get('name', 'unnamed')}")
    
    # Extract joint angles from JSON
    joints_dict = pose.get('joints', {})
    joint_values = [
        joints_dict.get('A1', 0),
        joints_dict.get('A2', 0),
        joints_dict.get('A3', 0),
        joints_dict.get('A4', 0),
        joints_dict.get('A5', 0),
        joints_dict.get('A6', 0),
        joints_dict.get('A7', 0)
    ]
    joint_angles = np.array(joint_values)
    
    # Extract KMR pose
    kmr_pose = pose.get('kmr_pose', {})
    robot_pos_x = kmr_pose.get('x', robot_pos_x) * 1000  # Convert from m to mm
    robot_pos_y = kmr_pose.get('y', robot_pos_y) * 1000  # Convert from m to mm
    robot_theta_base = kmr_pose.get('theta', robot_theta_base)
    
    print(f"Joint angles set to: {joint_angles}")
    print(f"Robot pose set to: x={robot_pos_x}, y={robot_pos_y}, theta={robot_theta_base}")

# List available poses
if handposes:
    list_poses()
    # Default to first pose
    set_pose_by_index(2)

# joint_angles = np.zeros(7) # Example: Zero angles

# --- NEW: Rotation for Manipulator Base relative to Mobile Base ---
# Rotate by -pi/2 around the Z-axis of the mounting point
theta_manip_base_rot_z = -np.pi / 2
c_m, s_m = np.cos(theta_manip_base_rot_z), np.sin(theta_manip_base_rot_z)
R_manip_local_z = np.array([[c_m, -s_m, 0],
                            [s_m,  c_m, 0],
                            [0,    0,   1]])


# --- Camera ---
T_ee_cam = np.array([
    [-0.01024101936856403, -0.9999474873612171, -0.0003795290256950858, 94.81681232855225],
    [0.9998571436962933,  -0.010234988685618118, -0.01345129012754058,    0.03266046476265938],
    [0.013446699289517097, -0.000517229730309641,  0.9999094552766381,   63.06449647669603],
    [0.0,                   0.0,                   0.0,                   1.0]
])

# --- Gripper ---
gripper_z_offset_ee = 210 # 21 cm = 210 mm in Z direction from EE frame

# Visualization parameters
axis_length_global = 500
axis_length_base = 300
axis_length_manipulator_base = 250
axis_length_joint = 100
axis_length_ee = 150
axis_length_camera = 80

# ==== Helper Functions ====

def inverse_homogeneous_transform(T):
    """Calculates the inverse of a 4x4 homogeneous transformation matrix."""
    # Option 1: Use numpy's built-in inverse function
    # return np.linalg.inv(T)
    
    # Option 2: More efficient method for homogeneous transforms (commented out)
    R = T[:3, :3]
    p = T[:3, 3]
    R_inv = R.T  # Inverse of rotation matrix is its transpose
    p_inv = -R_inv @ p # Inverse translation component
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = p_inv
    return T_inv

def transform_point(point_xyz, T):
    """Transforms a 3D point using a 4x4 homogeneous transformation matrix."""
    point_homogeneous = np.append(point_xyz, 1) # Convert to [x, y, z, 1]
    transformed_homogeneous = T @ point_homogeneous
    return transformed_homogeneous[:3] # Return [x', y', z']

def draw_basis(ax, origin, R, length, labels=True, colors=['r', 'g', 'b'], label_prefix=""):
    """Draws X, Y, Z basis vectors."""
    basis_vectors_local = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]])
    if R.shape == (4, 4):
        R = R[:3, :3]
    basis_vectors_global = R @ basis_vectors_local.T + origin[:3, np.newaxis]

    for i, (axis_name, color) in enumerate(zip(['X', 'Y', 'Z'], colors)):
        vec_start = origin[:3]
        vec_end = basis_vectors_global[:, i]
        ax.quiver(vec_start[0], vec_start[1], vec_start[2],
                  vec_end[0] - vec_start[0], vec_end[1] - vec_start[1], vec_end[2] - vec_start[2],
                  color=color, arrow_length_ratio=0.15, normalize=False)
        if labels:
             label_pos = vec_start + (vec_end - vec_start) * 1.15 # Increased offset slightly
             ax.text(label_pos[0], label_pos[1], label_pos[2], label_prefix+axis_name, color=color,
                     ha='center', va='center', fontsize=8)

def dh_matrix(a, d, alpha, theta):
    """Calculates the homogeneous transformation matrix from Standard DH parameters."""
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    cos_al, sin_al = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_th, -sin_th * cos_al,  sin_th * sin_al, a * cos_th],
        [sin_th,  cos_th * cos_al, -cos_th * sin_al, a * sin_th],
        [0,       sin_al,           cos_al,          d],
        [0,       0,                0,               1]
    ])

def forward_kinematics(dh_params, joint_angles):
    """
    Calculates the transformation matrices from the manipulator base to each joint frame.
    Returns a list of 4x4 matrices [T_base_j1, T_base_j2, ..., T_base_ee]
    Uses Standard DH convention: [a, d, alpha, theta_offset]
    """
    transforms = []
    T_cumulative = np.identity(4)
    if len(dh_params) != len(joint_angles):
        raise ValueError(f"Number of DH parameter sets ({len(dh_params)}) must match number of joint angles ({len(joint_angles)}).")

    for i in range(len(dh_params)):
        a, d, alpha, theta_offset = dh_params[i]
        # Calculate the effective theta for this joint
        theta = joint_angles[i] + theta_offset # Apply offset if provided
        T_i = dh_matrix(a, d, alpha, theta)
        T_cumulative = T_cumulative @ T_i
        transforms.append(T_cumulative.copy()) # Store a copy
    return transforms

# ==== Main Plotting ====
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Draw Global Basis (Origin at 0,0,0)
R_global = np.identity(3)
origin_global = np.array([0, 0, 0])
draw_basis(ax, origin_global, R_global, axis_length_global, labels=True, label_prefix="G ")

# 2. Define Mobile Base Transformation
origin_base_global = np.array([robot_pos_x, robot_pos_y, 0])
c_b, s_b = np.cos(robot_theta_base), np.sin(robot_theta_base)
R_base_global = np.array([[c_b, -s_b, 0],
                          [s_b,  c_b, 0],
                          [0,    0,   1]])
T_global_base = np.identity(4)
T_global_base[:3, :3] = R_base_global
T_global_base[:3, 3] = origin_base_global

# 3. Draw Mobile Base Cuboid
l2, w2, h = robot_length / 2, robot_width / 2, robot_height
vertices_local = np.array([
    [-l2, -w2, 0], [l2, -w2, 0], [l2, w2, 0], [-l2, w2, 0], # Bottom face
    [-l2, -w2, h], [l2, -w2, h], [l2, w2, h], [-l2, w2, h]  # Top face
])
vertices_global = (R_base_global @ vertices_local.T).T + origin_base_global
faces = [
    [vertices_global[0], vertices_global[1], vertices_global[2], vertices_global[3]], # Bottom
    [vertices_global[4], vertices_global[5], vertices_global[6], vertices_global[7]], # Top
    [vertices_global[0], vertices_global[1], vertices_global[5], vertices_global[4]], # Front (+X local)
    [vertices_global[2], vertices_global[3], vertices_global[7], vertices_global[6]], # Back (-X local)
    [vertices_global[1], vertices_global[2], vertices_global[6], vertices_global[5]], # Right (+Y local)
    [vertices_global[0], vertices_global[3], vertices_global[7], vertices_global[4]]  # Left (-Y local)
]
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.20))

# 4. Draw Mobile Base's Basis (at its origin)
draw_basis(ax, origin_base_global, R_base_global, axis_length_base, labels=False)

# 5. Calculate and Draw Manipulator Base
# Calculate manipulator origin relative to global frame
manipulator_base_origin_global = origin_base_global + R_base_global @ manipulator_base_offset_local

# Calculate manipulator orientation relative to global frame
# Apply mobile base rotation FIRST, then apply the manipulator's local rotation
R_manipulator_base_global = R_base_global @ R_manip_local_z # Apply the -pi/2 Z rotation

# Create the full homogeneous transformation matrix for the manipulator base
T_global_manipulator_base = np.identity(4)
T_global_manipulator_base[:3,:3] = R_manipulator_base_global
T_global_manipulator_base[:3, 3] = manipulator_base_origin_global

draw_basis(ax, manipulator_base_origin_global, R_manipulator_base_global, axis_length_manipulator_base, labels=True, label_prefix="M ")


obj_x, obj_y, obj_z = 1000, -500, 200
obj_theta_z = np.deg2rad(45)
c_obj, s_obj = np.cos(obj_theta_z), np.sin(obj_theta_z)

T_global_object = np.array([
        [-0.0319, 0.024, 0.9992, 12685.1941],
        [-0.9895, 0.1401, -0.035, 15230.1932],
        [-0.1409, -0.9898, 0.0193, 869.069],
        [0.0, 0.0, 0.0, 1.0]
    ])

T_manipulator_base_global = inverse_homogeneous_transform(T_global_manipulator_base)
T_manipulator_base_object = T_manipulator_base_global @ T_global_object

print("\n--- Object Pose Transformation ---")
print(f"Object Pose (Global Frame):\n{T_global_object.round(3)}")
print(f"Manipulator Base Pose (Global Frame):\n{T_global_manipulator_base.round(3)}")
print(f"Transform (Global -> Manipulator Base):\n{T_manipulator_base_global.round(3)}")
print(f"Object Pose (Manipulator Base Frame):\n{T_manipulator_base_object.round(3)}")

object_origin_global = T_global_object[:3, 3]
object_R_global = T_global_object[:3, :3]
object_axis_length = 120 # Choose a suitable length for the object's axes

draw_basis(ax, object_origin_global, object_R_global, object_axis_length,
           labels=True, label_prefix="Obj ", colors=['darkred', 'darkgreen', 'darkblue']) # Use different colors
ax.scatter(object_origin_global[0], object_origin_global[1], object_origin_global[2],
           c='red', marker='s', s=80, label='Object Origin (Global)') # Mark origin clearly


# 6. Perform Forward Kinematics for the Manipulator
# Uses the CORRECTED dh_params ordering now
transforms_base_to_joint = forward_kinematics(dh_params, joint_angles)

# 7. Draw Manipulator Joints and Links
joint_origins_global = [manipulator_base_origin_global]
transforms_global_to_joint = []

print("--- Joint Transformations (Global Frame) ---")
for i, T_base_joint in enumerate(transforms_base_to_joint):
    # Calculate global pose of the joint frame
    T_global_joint = T_global_manipulator_base @ T_base_joint
    transforms_global_to_joint.append(T_global_joint)

    joint_origin_global = T_global_joint[:3, 3]
    joint_R_global = T_global_joint[:3, :3]
    joint_origins_global.append(joint_origin_global)

    is_ee = (i == len(transforms_base_to_joint) - 1)
    if not is_ee:
        print(f"Joint {i+1} Pos: {joint_origin_global.round(2)}")
        draw_basis(ax, joint_origin_global, joint_R_global, axis_length_joint, labels=True, label_prefix=f"J{i+1} ")
    else:
        print(f"EE Pos: {joint_origin_global.round(2)}")
        draw_basis(ax, joint_origin_global, joint_R_global, axis_length_ee, labels=True, label_prefix="EE ")
        T_global_ee = T_global_joint

# Draw manipulator links
joint_origins_global_np = np.array(joint_origins_global)
ax.plot(joint_origins_global_np[:, 0], joint_origins_global_np[:, 1], joint_origins_global_np[:, 2], '-o', color='gray', linewidth=3, markersize=5, label='Manipulator Links')

# 8. Calculate and Draw Camera Frame
T_global_cam = T_global_ee @ T_ee_cam
cam_origin_global = T_global_cam[:3, 3]
cam_R_global = T_global_cam[:3, :3]
print(f"Cam Pos: {cam_origin_global.round(2)}")
draw_basis(ax, cam_origin_global, cam_R_global, axis_length_camera, labels=True, label_prefix="C ")
ax.scatter(cam_origin_global[0], cam_origin_global[1], cam_origin_global[2], c='black', marker='s', s=50, label='Camera Origin')

# 9. Calculate and Draw Gripper Point
p_ee_gripper = np.array([0, 0, gripper_z_offset_ee, 1])
p_global_gripper = T_global_ee @ p_ee_gripper
gripper_pos_global = p_global_gripper[:3]
print(f"Gripper Pos: {gripper_pos_global.round(2)}")
ax.scatter(gripper_pos_global[0], gripper_pos_global[1], gripper_pos_global[2], c='purple', marker='*', s=100, label='Gripper Point')

# ==== Plot Styling ====
# (Keep the dynamic limits and styling section from the previous version)
ax.set_xlabel("X Global (mm)")
ax.set_ylabel("Y Global (mm)")
ax.set_zlabel("Z Global (mm)")
ax.set_title("Robot + Manipulator Visualization (-90 deg base rot)")

# Set limits dynamically based on plotted objects
all_points = np.vstack([
    origin_global,
    origin_global + np.eye(3) * axis_length_global * 1.1, # Global axis tips
    vertices_global,
    np.array(joint_origins_global), # All joint origins including base and EE
    cam_origin_global,
    gripper_pos_global
])
# Include axis tips for EE and Camera for robust bounds
ee_origin_global = T_global_ee[:3, 3]
ee_R_global = T_global_ee[:3, :3]
ee_axis_tips = ee_origin_global[:,np.newaxis] + ee_R_global @ (np.eye(3) * axis_length_ee * 1.1)
all_points = np.vstack([all_points, ee_axis_tips.T])
cam_axis_tips = cam_origin_global[:,np.newaxis] + cam_R_global @ (np.eye(3) * axis_length_camera * 1.1)
all_points = np.vstack([all_points, cam_axis_tips.T])

min_coords = all_points.min(axis=0)
max_coords = all_points.max(axis=0)
range_coords = np.maximum(max_coords - min_coords, 100) * 1.2 # Add buffer, padding
mid_coords = (min_coords + max_coords) / 2
max_range = range_coords.max() # Make axes cubic based on max range


range = 1200

ax.set_xlim(robot_pos_x - range, robot_pos_x + range)
ax.set_ylim(robot_pos_y - range, robot_pos_y + range)
rangez1 = 2*range*0.1
rangez2 = 2*range*0.9
print(f"Z range: {-rangez1} to {rangez2}")
ax.set_zlim(-rangez1, rangez2)

ax.set_box_aspect([1,1,1]) # Enforce cubic aspect ratio

ax.legend()
plt.tight_layout()
plt.show()