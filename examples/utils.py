import numpy as np 

def infinite_horizon_lqr(A, B, Q, R, tol=1e-6, max_iter=1000):
    """
    Solves the infinite horizon LQR problem using a backward Riccati recursion.

    Args:
        A (ndarray): The state transition matrix.
        B (ndarray): The control input matrix.
        Q (ndarray): The state weighting matrix.
        R (ndarray): The control input weighting matrix.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.

    Returns:
        P (ndarray): The solution to the Riccati equation.
        K (ndarray): The optimal feedback gains.
    """
    n = A.shape[0]
    P = np.eye(n)  # Initialize P
    prev_P = np.zeros((n, n))

    for _ in range(max_iter):
        F = R + np.dot(B.T, np.dot(P, B))
        F_inv = np.linalg.inv(F)
        K = np.dot(F_inv, np.dot(B.T, np.dot(P, A)))
        PA_plus_ATP = np.dot(A.T, P) + np.dot(P, A)
        P_dot = PA_plus_ATP - np.dot(P, np.dot(B, np.dot(F_inv, np.dot(B.T, np.dot(P, A))))) + Q

        P -= P_dot  # Backward recursion

        if np.linalg.norm(P - prev_P) < tol:
            break

        prev_P = P.copy()

    return K, P

def quaternion_to_axis_angle(quaternion):
    """
    Convert a quaternion to an axis-angle representation as a single vector.
    The vector's direction represents the axis, and its magnitude represents the angle.

    Parameters:
    - quaternion: a tuple of four numbers (w, x, y, z), where w is the scalar and (x, y, z) is the vector part.

    Returns:
    - axis_angle_vector: a numpy array representing the axis of rotation and the angle as its magnitude.
    """

    # Extract the scalar and vector parts of the quaternion
    w, v = quaternion[0], np.array(quaternion[1:])

    # Compute the angle from the quaternion scalar part
    angle = 2 * np.arccos(w)

    # Calculate the axis from the quaternion vector part
    # Avoid division by zero in case of no rotation (when angle is 0)
    axis = v / np.sqrt(1 - w*w) if w != 1.0 else np.array([1, 0, 0])

    # Normalize the axis
    axis_normalized = axis / np.linalg.norm(axis)

    # Multiply the normalized axis by the angle to get the axis-angle vector
    axis_angle_vector = axis_normalized * angle

    return axis_angle_vector