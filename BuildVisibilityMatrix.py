import numpy as np

from Utils import *


def build_visibility_matrix(filtered_world_coords, filtered_feature_flags, camera_id):
    """
    This function builds the visibility matrix for the given camera_id.
    :param filtered_world_coords: World coordinates of the features that are visible in the current frame.
    :type filtered_world_coords: numpy.ndarray
    :param filtered_feature_flags: Feature flags of the features that are visible in the current frame.
    :type filtered_feature_flags: numpy.ndarray
    :param camera_id: Camera id.
    :type camera_id: int
    :return: X_index, visiblity_matrix
    :rtype: tuple
    """

    bin_temp = np.zeros((filtered_feature_flags.shape[0]), dtype = int)
    for n in range(camera_id + 1):
        bin_temp = bin_temp | filtered_feature_flags[:,n]

    X_index = np.where((filtered_world_coords.reshape(-1)) & (bin_temp))

    visiblity_matrix = filtered_world_coords[X_index].reshape(-1,1)
    for n in range(camera_id + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flags[X_index, n].reshape(-1,1)))

    o, c = visiblity_matrix.shape
    return X_index, visiblity_matrix[:, 1:c]