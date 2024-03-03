# Create Your Own Starter Code :)
# Authors: Hrishikesh Pawar, Tejas Rane
# Date: 02-11-2024
# Version: 1.0
# Description: This file contains the starter code for Phase 1 of the SFM project.

import os
import time
import argparse
import numpy as np

from natsort import natsorted

from Utils import *
from GetHomographyInliers import get_homography_inliers
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from GetInliersRANSAC import get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import get_essential_matrix
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation
from DisambiguateCameraPose import get_correct_camera_pose
from NonlinearTriangulation import nonlinear_triangulation
from LinearPnP import linear_pnp
from PnPRANSAC import PnPRANSAC
from NonlinearPnP import nonlinear_PnP
from BuildVisibilityMatrix import build_visibility_matrix
from BundleAdjustment import perform_bundle_adjustment


def run_sfm(data_path, log_dir="", no_log=False):
    """
    Main function for Phase 1 of the SFM project.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if not no_log:
        if log_dir: result_path = "Results/"+log_dir
        else: result_path = "Results/"+timestamp
        if not os.path.exists(result_path): os.makedirs(result_path)

    # Path to calibration text file
    calib_path = data_path + "calibration.txt"
    K = parse_text_file_to_K(calib_path)

    image_ids = []
    image_paths = []
    for file in natsorted(os.listdir(data_path)):
        if file.endswith(".png"):
            image_paths.append(data_path + file)
            image_ids.append(int(os.path.splitext(file)[0]))

    x_features, y_features, feature_flags = get_data(data_path, len(image_ids))
    filtered_feature_flags = np.zeros_like(feature_flags)

    all_combinations = [(a, b) for idx, a in enumerate(image_ids) for b in image_ids[idx + 1:]]

    inliers = {} # Dictionary of global inliers between all combinations
    structure = {} # Structure of the 3D world
    camera_poses = {} # Camera poses

    print(f"Outlier Rejection ...")

    # Reject outliers in all combinations
    for combination in all_combinations:
        image1_id, image2_id = combination
        combination_key = str(image1_id) + "_" + str(image2_id)

        print("Combination: ", combination_key)

        if not no_log:
            result_dir = result_path + "/" + combination_key
            if not os.path.exists(result_dir): os.makedirs(result_dir)

        _idx = np.where(feature_flags[:, image1_id-1] & feature_flags[:, image2_id-1])
        image1_coords_org = np.hstack((x_features[_idx, image1_id-1].reshape((-1,1)), y_features[_idx, image1_id-1].reshape((-1,1))))
        image2_coords_org = np.hstack((x_features[_idx, image2_id-1].reshape((-1,1)), y_features[_idx, image2_id-1].reshape((-1,1))))
        idx = np.array(_idx).reshape(-1)

        print('Number of matches Orginal: ', len(image1_coords_org))
        if not no_log:
            draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords_org, image2_coords_org,
                                 color=(0, 0, 255), save_path=result_dir+'/original_matches.png')

        """
        Estimate the fundamental matrix using the 8-point method
        """
        _, h_inlier_idx = get_homography_inliers(image1_coords_org, image2_coords_org, idx, threshold=30, n_max=1000)

        image1_coords = np.hstack((x_features[h_inlier_idx, image1_id-1].reshape((-1,1)), y_features[h_inlier_idx, image1_id-1].reshape((-1,1))))
        image2_coords = np.hstack((x_features[h_inlier_idx, image2_id-1].reshape((-1,1)), y_features[h_inlier_idx, image2_id-1].reshape((-1,1))))

        print('Number of matches Homography: ', len(image1_coords))
        if not no_log:
            draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords, image2_coords,
                                 color=(0, 255, 255), save_path=result_dir+'/homography_matches.png')

        F, f_inlier_idx = get_inliers_ransac(image1_coords, image2_coords, h_inlier_idx, threshold=0.06, n_max=1000)
        if combination_key == '1_2': F_12 = F

        image1_coords_inliers = np.hstack((x_features[f_inlier_idx, image1_id-1].reshape((-1,1)), y_features[f_inlier_idx, image1_id-1].reshape((-1,1))))
        image2_coords_inliers = np.hstack((x_features[f_inlier_idx, image2_id-1].reshape((-1,1)), y_features[f_inlier_idx, image2_id-1].reshape((-1,1))))

        print('Number of matches RANSAC: ', len(image1_coords_inliers))
        if not no_log:
            draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords_inliers, image2_coords_inliers,
                                 color=(0, 255, 0), save_path=result_dir+'/ransac_matches.png')

        inliers[combination_key] = [image1_coords_inliers, image2_coords_inliers]
        filtered_feature_flags[f_inlier_idx, image1_id-1] = 1
        filtered_feature_flags[f_inlier_idx, image2_id-1] = 1

    # Cosidering first 2 image pairs
    image1_id, image2_id = all_combinations[0]
    combination_key = str(image1_id) + "_" + str(image2_id)
    if not no_log: result_dir = result_path + "/" + combination_key

    print(f'Processing image pair: {combination_key}')

    _idx = np.where(filtered_feature_flags[:, image1_id-1] & filtered_feature_flags[:, image2_id-1])
    image1_inliers = np.hstack((x_features[_idx, 0].reshape((-1, 1)), y_features[_idx, 0].reshape((-1, 1))))
    image2_inliers = np.hstack((x_features[_idx, 1].reshape((-1, 1)), y_features[_idx, 1].reshape((-1, 1))))

    E = get_essential_matrix(F_12, K)

    C, R = extract_camera_pose(E)
    C0 = np.zeros(3)
    R0 = np.eye(3)

    camera_poses[image1_id] = [C0, R0]

    possible_world_coords = []

    for c, r in zip(C, R):
        _world_coords = linear_triangulation(K, C0, R0, c, r, image1_inliers, image2_inliers)
        possible_world_coords.append(_world_coords)

    if not no_log:
        plot_world_coords(possible_world_coords, save_path=result_dir+'/possible_world_coords.png')

    C_corr, R_corr = get_correct_camera_pose(C, R, possible_world_coords)
    camera_poses[image2_id] = [C_corr, R_corr]

    corr_world_coords = linear_triangulation(K, C0, R0, C_corr, R_corr, image1_inliers, image2_inliers)

    if not no_log:
        plot_world_coords([corr_world_coords], save_path=result_dir+'/corrected_world_coords.png', color='r', hold=True)

        draw_reprojections(image_paths[image1_id-1], image_paths[image2_id-1], K, C0, R0, C_corr, R_corr, corr_world_coords,
                           image1_inliers, image2_inliers, save_path=result_dir+'/corrected_reprojections.png')

    refined_world_coords = nonlinear_triangulation(K, C0, R0, C_corr, R_corr, image1_inliers, image2_inliers, corr_world_coords)

    if not no_log:
        plot_world_coords([refined_world_coords], save_path=result_dir+'/refined_world_coords.png', hold=True)

        plot_camera_pose(C0, R0, 1, save_path=result_dir+'/with_camera_pose.png', hold=True)
        plot_camera_pose(C_corr, R_corr, 2, save_path=result_dir+'/with_camera_pose.png', hold=True)

        draw_reprojections(image_paths[image1_id-1], image_paths[image2_id-1], K, C0, R0, C_corr, R_corr, refined_world_coords,
                           image1_inliers, image2_inliers, save_path=result_dir+'/refined_reprojections.png')

    print('Number of world points added: ', len(refined_world_coords))
    structure[combination_key] = refined_world_coords

    all_world_coords = np.zeros((x_features.shape[0], 3))
    all_world_coords_2 = np.zeros((x_features.shape[0], 3)) # plot
    cam_indices = np.zeros((x_features.shape[0], 1), dtype=int)
    filtered_world_coords = np.zeros((x_features.shape[0], 1), dtype=int)

    all_world_coords[_idx] = refined_world_coords
    all_world_coords_2[_idx] = refined_world_coords # plot
    filtered_world_coords[_idx] = 1
    cam_indices[_idx] = 1
    filtered_world_coords[np.where(all_world_coords[:2] < 0)] = 0

    C_set = []
    R_set = []
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C_corr)
    R_set.append(R_corr)

    for img_id in image_ids:
        if img_id == image1_id or img_id == image2_id:
            continue

        combination_key = str(image1_id) + "_" + str(img_id)
        if not no_log: result_dir = result_path + "/" + combination_key

        print('Processing image pair: ', combination_key)

        feature_idx_i = np.where(filtered_world_coords[:, 0] & filtered_feature_flags[:, img_id-1])
        if len(feature_idx_i[0]) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", img_id, "image")
            continue

        inliers_1 = np.hstack((x_features[feature_idx_i, image1_id-1].reshape(-1, 1), y_features[feature_idx_i, image1_id-1].reshape(-1, 1)))
        inliers_id = np.hstack((x_features[feature_idx_i, img_id-1].reshape(-1, 1), y_features[feature_idx_i, img_id-1].reshape(-1, 1)))
        world_coords = all_world_coords[feature_idx_i, :].reshape(-1, 3)

        C_new, R_new = PnPRANSAC(world_coords, inliers_id, K, threshold=200, n_max=1000)

        init_world_coords_new = linear_triangulation(K, C0, R0, C_new, R_new, inliers_1, inliers_id)
        corr_world_coords_new = nonlinear_triangulation(K, C0, R0, C_new, R_new, inliers_1, inliers_id, init_world_coords_new)
        # if not no_log:
        #     plot_world_coords([corr_world_coords_new], save_path=result_dir+'/world_coords_new.png', hold=True)

        C_new_corr, R_new_corr = nonlinear_PnP(K, C_new, R_new, inliers_id, corr_world_coords_new)
        C_new_corr, R_new_corr = nonlinear_PnP(K, C_new, R_new, inliers_id, world_coords)
        camera_poses[img_id] = [C_new_corr, R_new_corr]

        init_world_coords_new_corr = linear_triangulation(K, C0, R0, C_new_corr, R_new_corr, inliers_1, inliers_id)
        refined_world_coords_new = nonlinear_triangulation(K, C0, R0, C_new_corr, R_new_corr, inliers_1, inliers_id, init_world_coords_new_corr)
        if not no_log:
            plot_world_coords([refined_world_coords_new], save_path=result_dir+'/refined_world_coords_new.png', hold=True)

            plot_camera_pose(C_new_corr, R_new_corr, img_id, save_path=result_dir+'/with_camera_pose.png')

        R_set.append(R_new_corr)
        C_set.append(C_new_corr)

        for _img in range(1, img_id):
            world_coords_idx = np.where(filtered_feature_flags[:, _img-1] & filtered_feature_flags[:, img_id-1])
            world_coords_idx = np.squeeze(np.asarray(world_coords_idx))

            _key = str(_img) + "_" + str(img_id)
            if not no_log: result_dir = result_path + "/" + _key
            [C_img, R_img] = camera_poses[_img]

            if len(world_coords_idx) < 8:
                print("Got ", len(world_coords_idx), "common points between X and ", img_id, "image")
                continue

            _img1_inliers = np.hstack((x_features[world_coords_idx, _img-1].reshape(-1, 1), y_features[world_coords_idx, _img-1].reshape(-1, 1)))
            _img2_inliers = np.hstack((x_features[world_coords_idx, img_id-1].reshape(-1, 1), y_features[world_coords_idx, img_id-1].reshape(-1, 1)))

            _init_world_coords = linear_triangulation(K, C_img, R_img, C_new_corr, R_new_corr, _img1_inliers, _img2_inliers)
            _refined_world_coords = nonlinear_triangulation(K, C_img, R_img, C_new_corr, R_new_corr, _img1_inliers, _img2_inliers, _init_world_coords)
            if not no_log:
                plot_world_coords([_refined_world_coords], save_path=result_dir+'/refined_world_coords_.png', hold=True)

                plot_camera_pose(C_new_corr, R_new_corr, img_id, save_path=result_dir+'/with_camera_pose_.png', hold=True)

            print('Number of world points added: ', len(_refined_world_coords))

            all_world_coords[world_coords_idx] = _refined_world_coords
            all_world_coords_2[world_coords_idx] = _refined_world_coords
            filtered_world_coords[world_coords_idx] = 1

            print( 'Performing Bundle Adjustment  for image : ', img_id)
            R_set, C_set, all_world_coords = perform_bundle_adjustment(all_world_coords, filtered_world_coords, x_features, y_features,
                                                                        filtered_feature_flags, R_set, C_set, K, img_id-1)

            if not no_log:
                plot_world_coords([all_world_coords], save_path=result_dir+'/BA.png', hold=True)

                for i in range(img_id):
                    plot_camera_pose(C_set[i], R_set[i], i, save_path=result_dir+'/BA_with_camera_pose.png', hold=True)


    plot_world_coords([all_world_coords_2], save_path='before_BA.png', color='r', hold=True)
    plot_world_coords([all_world_coords], save_path='BA.png', color='b', hold=True)

    for i in range(img_id):
        plot_camera_pose(C_set[i], R_set[i], i, save_path='BA_with_camera_pose.png', hold=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Data/", help="Path of input images and feature matches text files")
    parser.add_argument("--log_dir", type=str, default="", help="Directory to save results")
    parser.add_argument("--no-log", action="store_true", help="Do not log results")
    args = parser.parse_args()

    run_sfm(args.data_path, args.log_dir, args.no_log)
