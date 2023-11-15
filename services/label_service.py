# Copyright (c) 2023, Sofya Vivdich and Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import numpy as np
import open3d as o3d

from utils.pcd_utils import build_map_wc
from utils.pcd_utils import color_pcd_by_labels
from utils.pcd_utils import get_close_point_indices
from utils.pcd_utils import get_subpcd
from utils.pcd_utils import get_visible_points


def get_map_not_zero_in_sphere(dataset, cam_name, start_index, end_index, R, visualize_steps=False, view_ind=1):
    # строим карту и отображение точек в инстансы
    map_wc = build_map_wc(dataset, cam_name, start_index, end_index)
    points2instances = build_points2instances_matrix(map_wc, dataset, cam_name, start_index, end_index)

    if visualize_steps:
        map_colored = color_pcd_by_labels(map_wc, points2instances[:, view_ind])
        o3d.visualization.draw_geometries([map_colored])

    # оставляем только те точки, которые размечены хотя бы на одном view
    not_zero_indices = get_not_zero_mask(points2instances)
    map_not_zero = get_subpcd(map_wc, not_zero_indices)
    points2instances_not_zero = points2instances[not_zero_indices]

    if visualize_steps:
        map_colored = color_pcd_by_labels(map_not_zero, points2instances_not_zero[:, view_ind])
        o3d.visualization.draw_geometries([map_colored])

    # оставляем только те точки, которые попали в сферу радиуса R с центром в start_index
    T_first_cam = dataset.get_lidar_pose(start_index) @ np.linalg.inv(dataset.get_camera_extrinsics(cam_name))
    close_point_indices = get_close_point_indices(map_not_zero, T_first_cam, R)
    map_final = get_subpcd(map_not_zero, close_point_indices)
    points2instances_final = points2instances_not_zero[close_point_indices]

    if visualize_steps:
        map_colored = color_pcd_by_labels(map_final, points2instances_final[:, view_ind])
        o3d.visualization.draw_geometries([map_colored])

    return map_final, points2instances_final


def build_points2instances_matrix(map_wc, dataset, cam_name, start_index, end_index):
    N = np.asarray(map_wc.points).shape[0]
    points2instances = np.zeros((N, end_index - start_index), dtype=int)

    for view_id, view in enumerate(range(start_index, end_index)):
        masks = dataset.get_image_instances(cam_name, view)
        image_labels = masks_to_image(masks)
        
        T = dataset.get_lidar_pose(view)
        T_cam = T @ np.linalg.inv(dataset.get_camera_extrinsics(cam_name))

        map_cc = copy.deepcopy(map_wc).transform(np.linalg.inv(T_cam)) # map in camera frame
        indices_visible = get_visible_points(map_cc)
        map_cc_visible = get_subpcd(map_cc, indices_visible)
        
        points_to_pixels = get_points_to_pixels(np.asarray(map_cc_visible.points), 
                                                dataset.get_camera_intrinsics(cam_name), 
                                                ((image_labels.shape[1], image_labels.shape[0])))

        for point_id, pixel_id in points_to_pixels.items():
            points2instances[indices_visible[point_id], view_id] = int(image_labels[pixel_id[1], pixel_id[0]])

    return points2instances


def get_points_to_pixels(points, cam_intrinsics, img_shape):
    img_width, img_height = img_shape

    points_proj = cam_intrinsics @ points.T
    points_proj[:2, :] /= points_proj[2, :]
    points_coord = points_proj.T

    inds = np.where(
        (points_coord[:, 0] < img_width) & (points_coord[:, 0] >= 0) &
        (points_coord[:, 1] < img_height) & (points_coord[:, 1] >= 0) &
        (points_coord[:, 2] > 0)
    )[0]

    points_ind_to_pixels = {}
    for ind in inds:
        points_ind_to_pixels[ind] = points_coord[ind][:2].astype(int)

    return points_ind_to_pixels


def get_not_zero_mask(points2instances):
    return np.any(points2instances != 0, axis=1)


def masks_to_image(masks):
        image_labels = np.zeros(masks[0]['segmentation'].shape)
        for i, mask in enumerate(masks):
            image_labels[mask['segmentation']] = i + 1
        return image_labels
