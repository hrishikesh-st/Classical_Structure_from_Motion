import numpy as np

from Utils import *

np.random.seed(69)


def find_homography(image1_coords, image2_coords):
    """
    Find the homography matrix between two images.

    :param image1_coords: Image 1 coordinates.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Image 2 coordinates.
    :type image2_coords: numpy.ndarray
    :return: Homography matrix.
    :rtype: numpy.ndarray
    """

    x1, y1, _ = zip(*image1_coords)
    x2, y2, _ = zip(*image2_coords)

    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)

    A = []

    for i in range(len(x1)):
        _x1, _y1, _x2, _y2 = x1[i], y1[i], x2[i], y2[i]
        A.append([_x1, _y1, 1, 0, 0, 0, -_x2*_x1, -_x2*_y1, -_x2])
        A.append([0, 0, 0, _x1, _y1, 1, -_y2*_x1, -_y2*_y1, -_y2])

    A = np.array(A)

    U, S, V = np.linalg.svd(A) # Single value decomposition
    H = np.reshape(V[-1], (3, 3))
    H /= H[2, 2]

    return H

def get_homography_inliers(image1_coords, image2_coords, indices, threshold, n_max=1000):
    """
    Get the inliers for the homography matrix.

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
    :return: Homography matrix, inlier indices
    :rtype: tuple
    """

    image1_coords_h = homogenize_coordinates(image1_coords)
    image2_coords_h = homogenize_coordinates(image2_coords)

    inlier_idx = []
    for _ in range(n_max):
        sampled_idx = np.random.choice(len(image1_coords_h), 8)
        sampled_i1_coords = [image1_coords_h[i] for i in sampled_idx]
        sampled_i2_coords = [image2_coords_h[i] for i in sampled_idx]

        H = find_homography(sampled_i1_coords, sampled_i2_coords)

        temp_inliers = []
        for i, (i1, i2) in enumerate(zip(image1_coords_h, image2_coords_h)):
            x1, y1, x2, y2 = i1[0], i1[1], i2[0], i2[1]
            _H = H.flatten().tolist()
            x2_hat = _H[0]*x1 + _H[1]*y1 + _H[2]
            y2_hat = _H[3]*x1 + _H[4]*y1 + _H[5]
            z2_hat = _H[6]*x1 + _H[7]*y1 + _H[8]

            if abs(x2_hat/(z2_hat+1e-6) - x2) + abs(y2_hat/(z2_hat+1e-6) - y2) < threshold:
                temp_inliers.append(indices[i]) # CHECK

        if len(temp_inliers) > len(inlier_idx):
            inlier_idx = temp_inliers

        if len(inlier_idx) >= 0.9*len(image1_coords):
            break

    return H, inlier_idx