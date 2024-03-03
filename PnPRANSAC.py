import numpy as np

from Utils import *
from LinearPnP import linear_pnp


def reprojection_error(world_coords, image_coords, C, R, K):
    """
    Compute reprojection error.

    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: 2D image coordinates.
    :type image_coords: numpy.ndarray
    :param C: Camera position.
    :type C: numpy.ndarray
    :param R: Rotation matrix.
    :type R: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Reprojection error.
    :rtype: float
    """
    T = np.hstack([np.eye(3), -C.reshape(-1, 1)])
    P = np.dot(K, np.dot(R, T))

    u_num = np.dot(P[0], world_coords.T)
    v_num = np.dot(P[1], world_coords.T)
    denom = np.dot(P[2], world_coords.T)

    u_ = u_num / denom
    v_ = v_num / denom

    error = (image_coords[0, 0] - u_)**2 + (image_coords[0, 1] - v_)**2
    return error

def PnPRANSAC(world_coords, image_coords, K, threshold, n_max=1000):
    """
    Perform PnP using RANSAC to estimate camera pose.

    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: 2D image coordinates.
    :type image_coords: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :param threshold: Threshold for inliers.
    :type threshold: float
    :param n_max: Maximum number of iterations for RANSAC.
    :type n_max: int, optional
    :return: Estimated camera position, estimated rotation matrix.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    inliers_idx = []
    for i in range(n_max):
        sampled_idx = np.random.choice(len(world_coords), 6)
        sampled_world_coords = world_coords[sampled_idx]
        sampled_image_coords = image_coords[sampled_idx]

        C, R = linear_pnp(sampled_world_coords, sampled_image_coords, K)

        temp_inliers = []

        for i, (X, x) in enumerate(zip(world_coords, image_coords)):
            error = reprojection_error(homogenize_coordinates(X.reshape(1, -1)), homogenize_coordinates(x.reshape(1, -1)), C, R, K)

            if error < threshold:
                temp_inliers.append(i)

        if len(temp_inliers) > len(inliers_idx):
            inliers_idx = temp_inliers

    inlier_world_coords = world_coords[inliers_idx]
    inlier_image_coords = image_coords[inliers_idx]

    C, R = linear_pnp(inlier_world_coords, inlier_image_coords, K)

    return C, R