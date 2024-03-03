import numpy as np

from Utils import *

def _normalize_coordinates(u, v):
    """
    Normalize the coordinates.

    :param u: u coordinates.
    :type u: numpy.ndarray
    :param v: v coordinates.
    :type v: numpy.ndarray
    :return: Normalized u and v coordinates, and the transformation matrix.
    :rtype: tuple
    """
    # Calculate mean and standard deviation of coordinates
    mean_u = np.mean(u)
    mean_v = np.mean(v)
    std_u = np.std(u)
    std_v = np.std(v)

    # Translation matrix to center the coordinates
    T = np.array([
        [1 / std_u, 0, -mean_u / std_u],
        [0, 1 / std_v, -mean_v / std_v],
        [0, 0, 1]
    ])

    # Apply the transformation to the coordinates
    u_normalized, v_normalized, _ = np.dot(T, np.column_stack((u, v, np.ones_like(u))).T)

    return u_normalized, v_normalized, T

def estimate_fundamental_matrix(image1_coords, image2_coords):
    """
    Estimate the fundamental matrix using the 8-point method.

    :param image1_coords: Coordinates of the points in the first image.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Coordinates of the points in the second image.
    :type image2_coords: numpy.ndarray
    :return: Fundamental matrix.
    :rtype: numpy.ndarray
    """
    # Extract the u and v coordinates of the matches
    # Matches is a list of tuples with the following format: (u1, v1, u2, v2)
    x1, y1, _ = zip(*image1_coords)
    x2, y2, _ = zip(*image2_coords)

    x1_n, y1_n, T1 = _normalize_coordinates(x1, y1)
    x2_n, y2_n, T2 = _normalize_coordinates(x2, y2)
    ones = np.ones((x1_n.shape[0]))

    A = [x1_n*x2_n, y1_n*x2_n, x2_n, x1_n*y2_n, y1_n*y2_n, y2_n, x1_n, y1_n, ones]
    A = np.vstack(A).T

    # Solve for the fundamental matrix using the 8-point method
    _, S, V = np.linalg.svd(A)
    f = V[np.argmin(S), :].reshape(3, 3)

    # Enforce rank-2 constraint on F
    Uf, Sf, Vf = np.linalg.svd(f)
    Sf[2] = 0
    F_n = Uf @ np.diag(Sf) @ Vf

    F = np.dot(T2.T, np.dot(F_n, T1))
    F /= F[2, 2]

    return F