import scipy
import numpy as np

# Following functions is taken from:
# https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html


def get_rotation_matrix(Q, type_ = 'q'):
    """
    Get rotation matrix from quaternion or rotation vector.

    :param Q: Quaternion or rotation vector.
    :type Q: numpy.ndarray
    :param type_: Type of input, 'q' for quaternion, 'e' for rotation vector.
    :type type_: str
    :return: Rotation matrix.
    :rtype: numpy.ndarray
    """
    if type_ == 'q':
        R = scipy.spatial.transform.Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = scipy.spatial.transform.Rotation.from_rotvec(Q)
        return R.as_matrix()

def get_euler_angles(R2):
    """
    Get Euler angles from rotation matrix.

    :param R2: Rotation matrix.
    :type R2: numpy.ndarray
    :return: Euler angles.
    :rtype: numpy.ndarray
    """
    euler = scipy.spatial.transform.Rotation.from_matrix(R2)
    return euler.as_rotvec()

def get_visibility_matrix(filtered_world_coords, filtered_feature_flags, camera_id):
    """
    Get visibility matrix.

    :param filtered_world_coords: Filtered world coordinates.
    :type filtered_world_coords: numpy.ndarray
    :param filtered_feature_flags: Filtered feature flags.
    :type filtered_feature_flags: numpy.ndarray
    :param camera_id: Camera ID.
    :type camera_id: int
    :return: Indices of visible world coordinates, visibility matrix.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    temp_bin = np.zeros((filtered_feature_flags.shape[0]), dtype=int)
    for n in range(camera_id + 1):
        temp_bin = temp_bin | filtered_feature_flags[:, n]

    world_coords_idx = np.where((filtered_world_coords.reshape(-1)) & (temp_bin))

    visiblity_matrix = filtered_world_coords[world_coords_idx].reshape(-1, 1)
    for n in range(camera_id+1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flags[world_coords_idx, n].reshape(-1, 1)))

    o, c = visiblity_matrix.shape
    return world_coords_idx, visiblity_matrix[:, 1:c]

def get_image_points(world_coords_idx, visiblity_matrix, x_features, y_features):
    """
    Get image points.

    :param world_coords_idx: Indices of visible world coordinates.
    :type world_coords_idx: numpy.ndarray
    :param visiblity_matrix: Visibility matrix.
    :type visiblity_matrix: numpy.ndarray
    :param x_features: X coordinates of features.
    :type x_features: numpy.ndarray
    :param y_features: Y coordinates of features.
    :type y_features: numpy.ndarray
    :return: Image points.
    :rtype: numpy.ndarray
    """
    pts2D = []
    visible_x_features = x_features[world_coords_idx]
    visible_y_features = y_features[world_coords_idx]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_x_features[i,j], visible_y_features[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)

def get_camera_point_indices(visiblity_matrix):
    """
    Get camera and point indices.

    :param visiblity_matrix: Visibility matrix.
    :type visiblity_matrix: numpy.ndarray
    :return: Camera indices, point indices.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

def bundle_adjustment_sparsity(filtered_world_coords, filtered_feature_flags, camera_id):
    """
    Generate sparsity pattern for bundle adjustment.

    :param filtered_world_coords: Filtered world coordinates.
    :type filtered_world_coords: numpy.ndarray
    :param filtered_feature_flags: Filtered feature flags.
    :type filtered_feature_flags: numpy.ndarray
    :param camera_id: Camera ID.
    :type camera_id: int
    :return: Sparsity matrix.
    :rtype: scipy.sparse.lil_matrix
    """
    number_of_cam = camera_id + 1
    X_index, visiblity_matrix = get_visibility_matrix(filtered_world_coords.reshape(-1), filtered_feature_flags, camera_id)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = scipy.sparse.lil_matrix((m, n), dtype=int)
    print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = get_camera_point_indices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (camera_id)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (camera_id) * 6 + point_indices * 3 + s] = 1

    return A


def project(points_3d, camera_params, K):
    """
    Project 3D points to 2D.

    :param points_3d: 3D points.
    :type points_3d: numpy.ndarray
    :param camera_params: Camera parameters.
    :type camera_params: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Projected points.
    :rtype: numpy.ndarray
    """
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = get_rotation_matrix(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """
    Rotate points using rotation vectors.

    :param points: Points to be rotated.
    :type points: numpy.ndarray
    :param rot_vecs: Rotation vectors.
    :type rot_vecs: numpy.ndarray
    :return: Rotated points.
    :rtype: numpy.ndarray
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def residual_function(x0, camera_id, n_points, camera_indices, point_indices, points_2d, K):
    """
    Residual function for optimization.

    :param x0: Initial guess for optimization.
    :type x0: numpy.ndarray
    :param camera_id: Camera ID.
    :type camera_id: int
    :param n_points: Number of points.
    :type n_points: int
    :param camera_indices: Indices of cameras.
    :type camera_indices: numpy.ndarray
    :param point_indices: Indices of points.
    :type point_indices: numpy.ndarray
    :param points_2d: 2D points.
    :type points_2d: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :return: Error vector.
    :rtype: numpy.ndarray
    """
    number_of_cam = camera_id + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()

    return error_vec

def perform_bundle_adjustment(all_world_coords, filtered_world_coords, x_features, y_features, filtered_feature_flags, R_set_, C_set_, K, camera_id):
    """
    Perform bundle adjustment.

    :param all_world_coords: All world coordinates.
    :type all_world_coords: numpy.ndarray
    :param filtered_world_coords: Filtered world coordinates.
    :type filtered_world_coords: numpy.ndarray
    :param x_features: X coordinates of features.
    :type x_features: numpy.ndarray
    :param y_features: Y coordinates of features.
    :type y_features: numpy.ndarray
    :param filtered_feature_flags: Filtered feature flags.
    :type filtered_feature_flags: numpy.ndarray
    :param R_set_: Set of rotation matrices.
    :type R_set_: numpy.ndarray
    :param C_set_: Set of camera positions.
    :type C_set_: numpy.ndarray
    :param K: Camera matrix.
    :type K: numpy.ndarray
    :param camera_id: Camera ID.
    :type camera_id: int
    :return: Optimized rotation matrices, optimized camera positions, optimized world coordinates.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    X_index, visiblity_matrix = get_visibility_matrix(filtered_world_coords, filtered_feature_flags, camera_id)
    points_3d = all_world_coords[X_index]
    points_2d = get_image_points(X_index, visiblity_matrix, x_features, y_features)

    RC_list = []
    for i in range(camera_id+1):
        C, R = C_set_[i], R_set_[i]
        Q = get_euler_angles(R)
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC_list.append(RC)
    RC_list = np.array(RC_list).reshape(-1, 6)

    x0 = np.hstack((RC_list.ravel(), points_3d.ravel()))
    n_points = points_3d.shape[0]

    camera_indices, point_indices = get_camera_point_indices(visiblity_matrix)

    A = bundle_adjustment_sparsity(filtered_world_coords, filtered_feature_flags, camera_id)

    res = scipy.optimize.least_squares(residual_function, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                                       args=(camera_id, n_points, camera_indices, point_indices, points_2d, K))

    x1 = res.x
    number_of_cam = camera_id + 1
    opt_camera_params = x1[:number_of_cam * 6].reshape((number_of_cam, 6))
    opt_points_3d = x1[number_of_cam * 6:].reshape((n_points, 3))

    opt_all_world_coords = np.zeros_like(all_world_coords)
    opt_all_world_coords[X_index] = opt_points_3d

    opt_C_set, opt_R_set = [], []
    for i in range(len(opt_camera_params)):
        R = get_rotation_matrix(opt_camera_params[i, :3], 'e')
        C = np.squeeze(opt_camera_params[i, 3:].reshape(1, 3))
        opt_C_set.append(C)
        opt_R_set.append(R)

    return opt_R_set, opt_C_set, opt_all_world_coords