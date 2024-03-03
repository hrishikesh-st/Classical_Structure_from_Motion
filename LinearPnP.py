import numpy as np

from Utils import *


def linear_pnp(world_coords, image_coords, K):
    """
    Linear PnP to estimate the camera pose.

    :param world_coords: World coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: Image coordinates.
    :type image_coords: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Camera center, Rotation matrix
    :rtype: tuple
    """

    A = []
    for i in range(len(world_coords)):
        A.append([world_coords[i, 0], world_coords[i, 1], world_coords[i, 2], 1, 0, 0, 0, 0,
                  -image_coords[i, 0]*world_coords[i, 0], -image_coords[i, 0]*world_coords[i, 1], -image_coords[i, 0]*world_coords[i, 2], -image_coords[i, 0]])
        A.append([0, 0, 0, 0, world_coords[i, 0], world_coords[i, 1], world_coords[i, 2], 1,
                  -image_coords[i, 1]*world_coords[i, 0], -image_coords[i, 1]*world_coords[i, 1], -image_coords[i, 1]*world_coords[i, 2], -image_coords[i, 1]])

    A = np.array(A)
    U, D, V = np.linalg.svd(A)
    P = V[-1, :].reshape(3, 4)

    R = np.linalg.inv(K) @ P[0:3, 0:3]
    Ur, Dr, Vr = np.linalg.svd(R)
    R = Ur @ Vr

    if np.linalg.det(R) < 0:
        R = -R

    C = -np.linalg.inv(K) @ P[:, 3]
    C /= Dr[0]

    return C, R