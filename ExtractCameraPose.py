import numpy as np

def extract_camera_pose(E):
    """
    Extracts the camera pose from the Essential Matrix.

    :param E: Essential Matrix.
    :type E: numpy.ndarray
    :return: Camera centers and rotation matrices.
    :rtype: tuple
    """

    C = []
    R = []

    U, _, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1 = U[:, 2]
    R1 = U @ W @ V_T
    if np.linalg.det(R1) == -1: R1 = -R1; C1 = -C1
    C.append(C1)
    R.append(R1)

    C2 = -U[:, 2]
    R2 = U @ W @ V_T
    if np.linalg.det(R2) == -1: R2 = -R2; C2 = -C2
    C.append(C2)
    R.append(R2)

    C3 = U[:, 2]
    R3 = U @ W.T @ V_T
    if np.linalg.det(R3) == -1: R3 = -R3; C3 = -C3
    C.append(C3)
    R.append(R3)

    C4 = -U[:, 2]
    R4 = U @ W.T @ V_T
    if np.linalg.det(R4) == -1: R4 = -R4; C4 = -C4
    C.append(C4)
    R.append(R4)

    return C, R