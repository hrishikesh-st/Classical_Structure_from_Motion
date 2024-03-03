import scipy
import numpy as np


def get_data(data_path, no_of_images):
    """
    Read data from matching files and extract features.

    :param data_path: Path to the directory containing matching files.
    :type data_path: str
    :param no_of_images: Number of images.
    :type no_of_images: int
    :return: x_features, y_features, feature_flags
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """   
    x_features = []
    y_features = []
    feature_flags = []

    for n in range(1, no_of_images):
        matching_file = open(data_path + "/matching" + str(n) + ".txt", "r")

        for i, row in enumerate(matching_file):
            if i == 0: continue # skip 1st line

            x_row = np.zeros((1, no_of_images))
            y_row = np.zeros((1, no_of_images))
            flag_row = np.zeros((1, no_of_images), dtype=int)
            row_elements = row.split()
            cols = [float(x) for x in row_elements]
            cols = np.asarray(cols)

            no_of_matches = cols[0]
            current_x = cols[4]
            current_y = cols[5]

            x_row[0, n-1] = current_x
            y_row[0, n-1] = current_y
            flag_row[0, n-1] = 1

            m = 1
            while no_of_matches > 1:
                image_id = int(cols[5+m])
                image_id_x = int(cols[6+m])
                image_id_y = int(cols[7+m])
                m += 3
                no_of_matches = no_of_matches - 1

                x_row[0, image_id-1] = image_id_x
                y_row[0, image_id-1] = image_id_y
                flag_row[0, image_id-1] = 1

            x_features.append(x_row)
            y_features.append(y_row)
            feature_flags.append(flag_row)

    x_features = np.asarray(x_features).reshape(-1, no_of_images)
    y_features = np.asarray(y_features).reshape(-1, no_of_images)
    feature_flags = np.asarray(feature_flags).reshape(-1, no_of_images)

    return x_features, y_features, feature_flags


def parse_text_file_to_K(file_path):
    """
    Parse text file to extract camera matrix K.

    :param file_path: Path to the text file.
    :type file_path: str
    :return: Camera matrix K.
    :rtype: numpy.ndarray
    """
    K = np.zeros((3, 3))

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        elements = line.split()
        K[i][0] = float(elements[0])
        K[i][1] = float(elements[1])
        K[i][2] = float(elements[2])

    return K


def homogenize_coordinates(x):
    """
    Homogenize coordinates.

    :param x: Coordinates.
    :type x: numpy.ndarray
    :return: Homogenized coordinates.
    :rtype: numpy.ndarray
    """
    return np.concatenate((x, np.ones((len(x), 1))), axis=1)


def unhomogenize_coordinates(x):
    """
    Unhomogenize coordinates.

    :param x: Homogenized coordinates.
    :type x: numpy.ndarray
    :return: Coordinates.
    :rtype: numpy.ndarray
    """
    return x[:, 0:2]


def convert_to_skew_matrix(x):
    """
    Convert vector to skew symmetric matrix.

    :param x: Vector.
    :type x: numpy.ndarray
    :return: Skew symmetric matrix.
    :rtype: numpy.ndarray
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def plot_world_coords(world_coords, save_path=None, color=None, hold=False):
    """
    Plot world coordinates.

    :param world_coords: World coordinates.
    :type world_coords: numpy.ndarray
    :param save_path: Path to save the plot.
    :type save_path: str, optional
    :param color: Color of the plot.
    :type color: str, optional
    :param hold: Whether to hold the plot or not.
    :type hold: bool, optional
    """
    import matplotlib.pyplot as plt

    for i, coord in enumerate(world_coords):
        c = np.array(coord)
        x = c[:, 0]
        y = c[:, 1]
        z = c[:, 2]
        plt.plot(x, z, '.', markersize=0.5, color=color)

    plt.axis([-20, 20, -10, 25])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if not hold: plt.close()
    else:
        plt.show()


def plot_camera_pose(C, R, id, save_path=None, color=None, hold=False):
    """
    Plot camera pose.

    :param C: Camera position.
    :type C: numpy.ndarray
    :param R: Rotation matrix.
    :type R: numpy.ndarray
    :param id: Camera id.
    :type id: int
    :param save_path: Path to save the plot.
    :type save_path: str, optional
    :param color: Color of the plot.
    :type color: str, optional
    :param hold: Whether to hold the plot or not.
    :type hold: bool, optional
    """
    import matplotlib.pyplot as plt

    angles = scipy.spatial.transform.Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    ax = plt.gca()
    ax.plot(C[0], C[2], marker=(3, 0, int(angles[1])), markersize=7)
    corr = -0.5
    ax.annotate(str(id+1), (C[0]+corr, C[2]+corr))
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if not hold: plt.close()
    else:
        plt.show()


def draw_features(image, corners, color=(0, 255, 0)):
    """
    Draw features on an image.

    :param image: Image.
    :type image: numpy.ndarray
    :param corners: Coordinates of corners.
    :type corners: numpy.ndarray
    :param color: Color of the features.
    :type color: tuple[int, int, int], optional
    :return: Image with features.
    :rtype: numpy.ndarray
    """
    import cv2

    for corner in corners:
        cv2.drawMarker(image, (int(corner[0]), int(corner[1])), color, cv2.MARKER_TILTED_CROSS, 10, 2)

    return image


def draw_feature_matches(image1, image2, image1_coords, image2_coords, save_path=None, color=(0, 255, 0)):
    """
    Draw feature matches between two images.

    :param image1: Path to image 1.
    :type image1: str
    :param image2: Path to image 2.
    :type image2: str
    :param image1_coords: Coordinates of features in image 1.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Coordinates of features in image 2.
    :type image2_coords: numpy.ndarray
    :param save_path: Path to save the output image.
    :type save_path: str, optional
    :param color: Color of the matches.
    :type color: tuple[int, int, int], optional
    """
    import cv2

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    image1_ = draw_features(image1, image1_coords, color=(255, 0, 0))
    image2_ = draw_features(image2, image2_coords, color=(255, 0, 0))

    image1_coords_ = [cv2.KeyPoint(i[0], i[1], 1) for i in image1_coords]
    image2_coords_ = [cv2.KeyPoint(i[0], i[1], 1) for i in image2_coords]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(image1_coords))]

    output_img = cv2.drawMatches(image1_, image1_coords_, image2_, image2_coords_, matches, None, matchColor=color, flags=2)

    # Display the output image
    if save_path:
        cv2.imwrite(save_path, output_img)
    else:
        cv2.imshow('Matches', output_img)
        cv2.waitKey(0)


def draw_reprojections(image1, image2, K, C1, R1, C2, R2, world_coords, image1_coords, image2_coords, save_path=None):
    """
    Draw reprojections of 3D points onto two images.

    :param image1: Path to image 1.
    :type image1: str
    :param image2: Path to image 2.
    :type image2: str
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
    :param world_coords: 3D world coordinates.
    :type world_coords: numpy.ndarray
    :param image1_coords: Coordinates of features in image 1.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Coordinates of features in image 2.
    :type image2_coords: numpy.ndarray
    :param save_path: Path to save the output image.
    :type save_path: str, optional
    """
    import cv2

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    image1_ = draw_features(image1, image1_coords, color=(0, 0, 255))
    image2_ = draw_features(image2, image2_coords, color=(0, 0, 255))

    T1 = np.hstack([np.eye(3), np.array([-C1]).T])
    P1 = np.dot(K, np.dot(R1, T1))

    image1_coords_p = np.matmul(P1, homogenize_coordinates(world_coords).T).T
    image1_coords_p = image1_coords_p/image1_coords_p[:, 2].reshape(-1, 1)

    T2 = np.hstack([np.eye(3), np.array([-C2]).T])
    P2 = np.dot(K, np.dot(R2, T2))

    image2_coords_p = np.matmul(P2, homogenize_coordinates(world_coords).T).T
    image2_coords_p = image2_coords_p/image2_coords_p[:, 2].reshape(-1, 1)

    image1_p = draw_features(image1_, image1_coords_p[:, 0:2], color=(0, 255, 0))
    image2_p = draw_features(image2_, image2_coords_p[:, 0:2], color=(0, 255, 0))

    image = np.concatenate((image1_p, image2_p), axis=1)

    if save_path:
        cv2.imwrite(save_path, image)
    else:
        cv2.imshow("Reprojection", image)
        cv2.waitKey(0)
