import scipy
import numpy as np

from Utils import *


def reprojection_error(world_coords, image_coords, C, R, K):
    """
    Compute Reprojection error.

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
    :rtype: numpy.ndarray
    """

    T = np.hstack([np.eye(3), -C.reshape(-1, 1)])
    P = np.dot(K, np.dot(R, T))

    u_num = np.dot(P[0], world_coords.T)
    v_num = np.dot(P[1], world_coords.T)
    denom = np.dot(P[2], world_coords.T)

    u_ = u_num / denom
    v_ = v_num / denom

    error = np.concatenate(((image_coords[:, 0] - u_)**2, (image_coords[:, 1] - v_)**2))
    return error

def residual_function(x0, world_coords, image_coords, K):
    """
    Residual function for the Nonlinear PnP.

    :param x0: Initial guess for optimization.
    :type x0: numpy.ndarray
    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: 2D image coordinates.
    :type image_coords: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Error vector.
    :rtype: numpy.ndarray
    """

    c0 = x0[0:3]
    r0 = scipy.spatial.transform.Rotation.from_quat(x0[3:]).as_matrix()
    error = reprojection_error(homogenize_coordinates(world_coords), homogenize_coordinates(image_coords), c0, r0, K)
    return error

def nonlinear_PnP(K, C, R, image_coords, world_coords):
    """
    Nonlinear PnP algorithm to estimate camera pose.

    :param K: Camera matrix.
    :type K: numpy.ndarray
    :param C: Initial estimate of camera position.
    :type C: numpy.ndarray
    :param R: Initial estimate of rotation matrix.
    :type R: numpy.ndarray
    :param image_coords: 2D image coordinates.
    :type image_coords: numpy.ndarray
    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :return: Optimized camera position, optimized rotation matrix.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    Rq = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()

    init_camera_pose = np.concatenate((C, Rq)).flatten()
    optimized_result = scipy.optimize.least_squares(residual_function, x0=init_camera_pose, method='lm',
                                                    args=(world_coords, image_coords, K))

    C_opt = optimized_result.x[0:3]
    R_opt = scipy.spatial.transform.Rotation.from_quat(optimized_result.x[3:]).as_matrix()

    return C_opt, R_opt