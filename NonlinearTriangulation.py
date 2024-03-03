import scipy
import numpy as np

from Utils import *


def reprojection_error(world_coords, image_coords, P):
    """
    Compute reprojection error.

    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :param image_coords: 2D image coordinates.
    :type image_coords: numpy.ndarray
    :param P: Projection matrix.
    :type P: numpy.ndarray
    :return: Reprojection error.
    :rtype: numpy.ndarray
    """
    projected_coords = np.matmul(P, world_coords)
    projected_coords = projected_coords/projected_coords[2]
    error = image_coords - projected_coords.T
    return error ** 2

def residual_function(x0, P1, P2, image1_coords, image2_coords, idx):
    """
    Residual function for nonlinear triangulation.

    :param x0: Initial guess for optimization.
    :type x0: numpy.ndarray
    :param P1: Projection matrix for image 1.
    :type P1: numpy.ndarray
    :param P2: Projection matrix for image 2.
    :type P2: numpy.ndarray
    :param image1_coords: 2D image coordinates for image 1.
    :type image1_coords: numpy.ndarray
    :param image2_coords: 2D image coordinates for image 2.
    :type image2_coords: numpy.ndarray
    :param idx: Index of the point.
    :type idx: int
    :return: Total error vector.
    :rtype: numpy.ndarray
    """
    error1 = reprojection_error(x0.reshape((-1, 1)), image1_coords[idx], P1)
    error2 = reprojection_error(x0.reshape((-1, 1)), image2_coords[idx], P2)
    total_error = np.concatenate((error1[0, :2], error2[0, :2]))

    return total_error

def nonlinear_triangulation(K, C1, R1, C2, R2, image1_coords, image2_coords, init_world_coords):
    """
    Perform nonlinear triangulation to refine 3D world coordinates.

    :param K: Camera matrix.
    :type K: numpy.ndarray
    :param C1: Camera position for image 1.
    :type C1: numpy.ndarray
    :param R1: Rotation matrix for image 1.
    :type R1: numpy.ndarray
    :param C2: Camera position for image 2.
    :type C2: numpy.ndarray
    :param R2: Rotation matrix for image 2.
    :type R2: numpy.ndarray
    :param image1_coords: 2D image coordinates for image 1.
    :type image1_coords: numpy.ndarray
    :param image2_coords: 2D image coordinates for image 2.
    :type image2_coords: numpy.ndarray
    :param init_world_coords: Initial estimated 3D world coordinates.
    :type init_world_coords: numpy.ndarray
    :return: Refined 3D world coordinates.
    :rtype: numpy.ndarray
    """
    image1_coords_h = homogenize_coordinates(image1_coords)
    image2_coords_h = homogenize_coordinates(image2_coords)
    init_world_coords_h = homogenize_coordinates(init_world_coords)

    T1 = np.hstack([np.eye(3), np.array([-C1]).T])
    P1 = np.dot(K, np.dot(R1, T1))

    T2 = np.hstack([np.eye(3), np.array([-C2]).T])
    P2 = np.dot(K, np.dot(R2, T2))

    refined_world_coords = []

    for i, x in enumerate(init_world_coords_h):
        optimized_result = scipy.optimize.least_squares(residual_function, x0=x, method='lm',
                                                        args=(P1, P2, image1_coords_h, image2_coords_h, i))
        _coords = optimized_result.x
        refined_world_coords.append(_coords/_coords[3])

    refined_world_coords = np.vstack(refined_world_coords)

    return refined_world_coords[:, 0:3]
