import numpy as np
import math


def get_rotation_matrix_x(angle: float = 0.0) -> np.array:
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(angle), -np.sin(angle), 0],
                       [0, np.sin(angle), np.cos(angle), 0],
                       [0, 0, 0, 1]])

        return Rx


def get_rotation_matrix_y(angle: float = 0.0) -> np.array:
    Ry = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1]])

    return Ry


def get_rotation_matrix_z(angle: float = 0.0) -> np.array:
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    return Rz


def rotation_matrix_to_zyx_extrinsic(R):
    """
    Extracts Euler angles (alpha, beta, gamma) for ZYX extrinsic convention
    from a 3x3 rotation matrix R.

    The matrix is assumed to be constructed as R = Rz(alpha) @ Ry(beta) @ Rx(gamma).

    Args:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: (alpha, beta, gamma) angles in radians.
               alpha: rotation around Z
               beta: rotation around Y
               gamma: rotation around X
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"

    # Calculate beta (rotation around Y)
    # sin(beta) = -R[2, 0]
    # Using arcsin, handle potential numerical inaccuracies near +/- 1
    sin_beta = -R[2, 0]
    if sin_beta > 1.0:
        sin_beta = 1.0
    elif sin_beta < -1.0:
        sin_beta = -1.0
    
    beta = math.asin(sin_beta) # Gives range [-pi/2, pi/2]

    cos_beta = math.cos(beta)

    # Check for gimbal lock (cos(beta) close to 0)
    epsilon = 1e-6  # Threshold for near-zero cosine

    if abs(cos_beta) > epsilon:
        # Not in gimbal lock
        # alpha = atan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
        alpha = math.atan2(R[1, 0], R[0, 0])

        # gamma = atan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
        gamma = math.atan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock
        gamma = 0.0 # Conventionally set gamma to 0

        # Determine alpha based on the lock configuration
        if beta > 0: # beta is close to pi/2 (sin_beta approx 1, R[2,0] approx -1)
             # alpha - gamma = atan2(-r12, r13) -> alpha = atan2(-R[0,1], R[0,2])
            alpha = math.atan2(-R[0, 1], R[0, 2])
        else: # beta is close to -pi/2 (sin_beta approx -1, R[2,0] approx 1)
            # alpha + gamma = atan2(-r12, -r13) -> alpha = atan2(-R[0,1], -R[0,2])
            alpha = math.atan2(-R[0, 1], -R[0, 2]) # Note the sign difference

    return alpha, beta, gamma

def rotation_matrix_to_xyz_extrinsic(R):
    """
    Extracts Euler angles (alpha, beta, gamma) for XYZ extrinsic convention
    from a 3x3 rotation matrix R.

    The matrix is assumed to be constructed as R = Rx(alpha) @ Ry(beta) @ Rz(gamma).

    Args:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: (alpha, beta, gamma) angles in radians.
               alpha: rotation around X
               beta: rotation around Y
               gamma: rotation around Z
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"

    # Calculate beta (rotation around Y)
    # sin(beta) = -R[2, 0]
    # Using arcsin, handle potential numerical inaccuracies near +/- 1
    sin_beta = -R[2, 0]
    if sin_beta > 1.0:
        sin_beta = 1.0
    elif sin_beta < -1.0:
        sin_beta = -1.0
    
    beta = math.asin(sin_beta) # Gives range [-pi/2, pi/2]

    cos_beta = math.cos(beta)

    # Check for gimbal lock (cos(beta) close to 0)
    epsilon = 1e-6  # Threshold for near-zero cosine

    if abs(cos_beta) > epsilon:
        # Not in gimbal lock
        # alpha = atan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
        alpha = math.atan2(R[1, 0], R[0, 0])

        # gamma = atan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
        gamma = math.atan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock
        gamma = 0.0 # Conventionally set gamma to 0

        # Determine alpha based on the lock configuration
        if beta > 0: # beta is close to pi/2 (sin_beta approx 1, R[2,0] approx -1)
             # alpha - gamma = atan2(-r12, r13) -> alpha = atan2(-R[0,1], R[0,2])
            alpha = math.atan2(-R[0, 1], R[0, 2])
        else: # beta is close to -pi/2 (sin_beta approx -1, R[2,0] approx 1)
            # alpha + gamma = atan2(-r12, -r13) -> alpha = atan2(-R[0,1], -R[0,2])
            alpha = math.atan2(-R[0, 1], -R[0, 2])

    return alpha, beta, gamma

    # return gamma, beta, alpha