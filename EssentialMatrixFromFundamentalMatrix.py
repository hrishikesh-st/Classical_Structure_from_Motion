import numpy as np

def get_essential_matrix(F, K):
    """
    Compute Essential Matrix from the Fundamental Matrix and the Camera Matrix.

    :param F: Fundamental Matrix.
    :type F: numpy.ndarray
    :param K: Camera Matrix.
    :type K: numpy.ndarray
    :return: Essential Matrix.
    :rtype: numpy.ndarray
    """

    _E = K.T @ F @ K # Essential Matrix
    U, _, V_T = np.linalg.svd(_E)  # Singular Value Decomposition

    D = np.diag([1, 1, 0])

    E = U @ D @ V_T # Corrected Essential Matrix

    return E