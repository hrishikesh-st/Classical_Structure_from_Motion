import numpy as np

from Utils import *

def get_correct_camera_pose(C, R, X):
    """
    Disambiguates the correct camera pose.

    :param C: Camera center.
    :type C: numpy.ndarray
    :param R: Rotation matrix.
    :type R: numpy.ndarray
    :param X: World coordinates.
    :type X: numpy.ndarray
    :return: C_corr, R_corr
    :rtype: tuple
    """

    max_inliers = 0
    for c, r, x in zip(C, R, X):

        r3 = r[:, 2]
        _c = c.reshape((3, 1))

        cheirality = r3.T @ (x.T - _c)
        inliers = np.where(cheirality > 0)

        inliers = np.logical_and(cheirality > 0, x[:, 2] > 0)

        if np.sum(inliers) >= max_inliers:
            C_corr = c
            R_corr = r
            max_inliers = np.sum(inliers)

    return C_corr, R_corr