import numpy as np

from Utils import *
from EstimateFundamentalMatrix import estimate_fundamental_matrix

np.random.seed(69)


def get_inliers_ransac(image1_coords, image2_coords, indices, threshold, n_max=1000):
    """
    Get the inliers using the RANSAC algorithm.

    :param image1_coords: Coordinates of the points in the first image.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Coordinates of the points in the second image.
    :type image2_coords: numpy.ndarray
    :param indices: Indices of the points.
    :type indices: numpy.ndarray
    :param threshold: Error threshold.
    :type threshold: float
    :param n_max: Maximum number of iterations, defaults to 1000 
    :type n_max: int, optional
    :return: Fundamental matrix, inlier indices
    :rtype: tuple
    """

    image1_coords_h = homogenize_coordinates(image1_coords)
    image2_coords_h = homogenize_coordinates(image2_coords)

    inliers_idx = []
    for _ in range(n_max):
        sampled_idx = np.random.choice(len(image1_coords_h), 8)
        sampled_i1_coords = [image1_coords_h[i] for i in sampled_idx]
        sampled_i2_coords = [image2_coords_h[i] for i in sampled_idx]

        _F = estimate_fundamental_matrix(sampled_i1_coords, sampled_i2_coords)

        temp_inliers = []

        for i, (i1, i2) in enumerate(zip(image1_coords_h, image2_coords_h)):
            error = i2 @ _F @ i1.T

            if abs(error) < threshold:
                temp_inliers.append(indices[i]) # CHECK

        if len(temp_inliers) > len(inliers_idx):
            inliers_idx = temp_inliers
            F = _F

        # if len(inliers_idx) >= 0.95*len(image1_coords):
        #     break


    return F, inliers_idx