import numpy as np

from Utils import *

def linear_triangulation(K, C1, R1, C2, R2, image1_coords, image2_coords):
    """
    Linear Triangulation to estimate the 3D world coordinates.

    :param K: Camera matrix.
    :type K: numpy.ndarray
    :param C1: Camera 1 position.
    :type C1: numpy.ndarray
    :param R1: Camera 1 rotation matrix.
    :type R1: numpy.ndarray
    :param C2: Camera 2 position.
    :type C2: numpy.ndarray
    :param R2: Camera 2 rotation matrix.
    :type R2: numpy.ndarray
    :param image1_coords: Image 1 coordinates.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Image 2 coordinates.
    :type image2_coords: numpy.ndarray
    :return: 3D world coordinates.
    :rtype: numpy.ndarray
    """

    world_coords = []

    image1_coords_h = homogenize_coordinates(image1_coords)
    image2_coords_h = homogenize_coordinates(image2_coords)

    T1 = np.hstack([np.eye(3), np.array([-C1]).T])
    P1 = np.dot(K, np.dot(R1, T1))

    T2 = np.hstack([np.eye(3), np.array([-C2]).T])
    P2 = np.dot(K, np.dot(R2, T2))

    for i1, i2 in zip(image1_coords_h, image2_coords_h):
        A1 = convert_to_skew_matrix(i1) @ P1
        A2 = convert_to_skew_matrix(i2) @ P2
        A = np.vstack((A1, A2))

        _, S, V = np.linalg.svd(A)
        X = V[np.argmin(S), :]
        X = X/X[3]

        world_coords.append(X[0:3])

    world_coords = np.vstack(world_coords)
    return world_coords