from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from KMR_communication import utils
from KMR_communication import config


def plot_object_pose():
        
    # Standard basis vectors
    standard_basis = np.eye(3)

    # Given basis
    given_basis = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    given_basis = np.array([
        [-0.6149, 0.01, 0.7886],
        [0.7886, -0.0026, 0.6149],
        [0.0082, 0.9999, -0.0063]
    ])
    given_basis = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

    given_basis = given_basis.T

    camera_basis = np.array([
        [0.2352259475716086, -0.12479936957041475, -0.9638951555765105],
        [0.9718930667607659, 0.040021878517851536, 0.2319959396676064],
        [0.009623947807015322, -0.991374483530179, 0.13070582631955796]
    ])
    camera_basis = camera_basis.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot standard basis
    for vec, color in zip(standard_basis, ['r', 'g', 'b']):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, label=f'Standard {vec}')

    # Plot given basis
    for vec, color in zip(given_basis, ['r', 'g', 'b']):
        ax.quiver(1, 1, 1, vec[0], vec[1], vec[2], color=color, linestyle='dashed', label=f'Given {vec}')

    for vec, color in zip(camera_basis, ['r', 'g', 'b']):
        ax.quiver(3, 1, 1, vec[0], vec[1], vec[2], color=color, linestyle='dotted', label=f'Camera {vec}')



    # Set viewing angle (elevation and azimuth)
    ax.view_init(elev=39, azim=108)

    limits = 3
    # Set plot limits
    ax.set_xlim([-limits, limits])
    ax.set_ylim([-limits, limits])
    ax.set_zlim([-limits, limits])

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend()

    plt.show()


def calculate_correct_rotation():
    object_in_camera_1 = np.array([
        [-0.0832, -0.9878, -0.1315, -106.8358],
        [-0.9964, 0.0843, -0.003, 42.3535],
        [0.014, 0.1308, -0.9913, 309.481],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    object_in_camera_2 = np.array([
        [0.9999, 0.0004, -0.0103, 4.825],
        [-0.0103, -0.004, -0.9999, 54.5016],
        [-0.0005, 1., -0.004, 315.1496],
        [0., 0., 0., 1.]
    ])

    iiwa_pos_1 = {
        "A": -0.0002153650448372649,
        "B": -0.00014246950139413654,
        "C": -3.099917071663849,
        "z": 448.5953807247249,
        "y": 790.0026230495795,
        "x": 9.61347822940834
    }

    kmr_pose= {
        "theta": -3.1402262724805574,
        "y": 15.054806942806115,
        "x": 14.009484855858737
    }

    iiwa_pos_2 = {
        "A": 0.8698737404552603,
        "B": -1.5575229240214776,
        "C": -2.3791159088026723,
        "z": 44.228709668057206,
        "y": 537.4164697012388,
        "x": 85.79211483685623
    }



    def calclulate_object_in_world_from_poses(kmr_pose, iiwa_pos, T_cam_obj):
        T_world_ee = utils.calculate_end_effector_in_world(kmr_pose, iiwa_pos)
        extrinsic_data = utils.load_json_data(config.CAMERA_EXTRINSIC_FILE)
        T_ee_cam = np.array(extrinsic_data["transformation_matrix"])
        T_world_cam = T_world_ee @ T_ee_cam
        print("T_cam_obj: \n", T_cam_obj)
        T_cam_obj[0:3, 0:3] = T_cam_obj[0:3, 0:3] @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T_cam_obj[0, 3] = T_cam_obj[0, 3]
        T_cam_obj[1, 3] = T_cam_obj[1, 3]
        T_cam_obj[2, 3] = T_cam_obj[2, 3]
        print("T_cam_obj: \n", T_cam_obj)

        T_world_obj = utils.calculate_object_in_world(T_world_cam, T_cam_obj)
        return T_world_obj

    T_world_obj_1 = calclulate_object_in_world_from_poses(kmr_pose, iiwa_pos_1, object_in_camera_1)
    print("T_world_obj: \n", T_world_obj_1)

    print("\n"*3)

    T_world_obj_2 = calclulate_object_in_world_from_poses(kmr_pose, iiwa_pos_2, object_in_camera_2)
    print("T_world_obj: \n", T_world_obj_2)
    

    


    return

def rotx(theta):
    """Rotation about x-axis by theta radians."""
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
def roty(theta):
    """Rotation about y-axis by theta radians."""
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
def rotz(theta):
    """Rotation about z-axis by theta radians."""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])
    

def Grabing_rotations():
    # Get user input for pose name and rotation axis
    name = "down"
    rotation_axis = "z"
    # Original 4x4 matrices
    matrix1 = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 10],
        [0, 0, 1, -234],
        [0, 0, 0, 1]
    ], dtype=float)

    matrix2 = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 10],
        [0, 0, 1, -310],
        [0, 0, 0, 1]
    ], dtype=float)

    # Define the rotation function based on the selected axis
    if rotation_axis == 'x':
        rotation_func = rotx
    elif rotation_axis == 'y':
        rotation_func = roty
    else:  # Default to z-axis
        rotation_func = rotz

    # Generate 4 rotations in 90-degree increments
    for i in range(4):
        # Calculate rotation angle in radians (90 * i degrees)
        angle = np.radians(90 * i)
        
        # Create rotation matrix
        rotation_matrix = rotation_func(angle)
        
        # Create 4x4 transformation matrix from 3x3 rotation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        
        # Apply rotation to both matrices
        rotated_matrix1 = transform @ matrix1
        rotated_matrix2 = transform @ matrix2

        
        # print(rotated_matrix1)


        # Print in the requested format
        print("    {")
        print(f'        "grabbing_pose_name": "{name}_{i+1}",')
        print('        "T_object_ee_grasp": [')
        for row in rotated_matrix1:
            print("            [")
            print("                " + ",\n                ".join(f"{val:.1f}" for val in row))
            print("            ]" + ("," if not np.array_equal(row, rotated_matrix1[-1]) else ""))
        print("        ],")
        print('        "T_object_ee_before_grasp": [')
        for row in rotated_matrix2:
            print("            [")
            print("                " + ",\n                ".join(f"{val:.1f}" for val in row))
            print("            ]" + ("," if not np.array_equal(row, rotated_matrix2[-1]) else ""))
        print("        ]")
        print("    }," if i < 3 else "    }")
        print("")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    Grabing_rotations()