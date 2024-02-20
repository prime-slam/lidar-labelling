# Copyright (c) 2023, Sofia Vivdich and Anastasiia Kornilova
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

from src.utils.image_utils import generate_random_colors


def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def get_subpcd(pcd, indices):
    subpcd = o3d.geometry.PointCloud()
    subpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    return subpcd


def remove_statistical_outlier_points(pcd, nb_neighbors=25, std_ratio=5.0):
    pcd_copy = copy.deepcopy(pcd)
    pcd_result, indices = pcd_copy.remove_statistical_outlier(nb_neighbors, std_ratio)
    return pcd_result, indices


def build_map_wc_triangle_mesh(
    dataset, cam_name, start_index, end_index, visualize=False
):
    map_wc = o3d.geometry.PointCloud()

    geometries = []
    for i in range(start_index, end_index):
        T = dataset.get_lidar_pose(i)
        # Shift the cloud to the world coordinate system
        map_wc += copy.deepcopy(dataset.get_point_cloud(i)).transform(T)

        m = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # Camera position in the world coordinate system
        T_cam = T @ np.linalg.inv(dataset.get_camera_extrinsics(cam_name))
        geometries.append(m.transform(T_cam))

    if visualize:
        # Visualize the map and the location of the camera relative to it
        o3d.visualization.draw_geometries([map_wc] + geometries)

    return map_wc, geometries


def get_visible_points(pcd_centered, visualize=False):
    """Removing hidden points

    Parameters
    ----------
    pcd_centered : open3d.geometry.PointCloud
        pcd in the camera coordinate system
    visualize : boolean
        true, if visualization of the result is required.
        false, otherwise
    """

    diameter = np.linalg.norm(
        np.asarray(pcd_centered.get_max_bound())
        - np.asarray(pcd_centered.get_min_bound())
    )

    # In this algorithm, the pcd is considered in the camera coordinate system,
    # so the camera position is defined as the origin of coordinates
    camera = [0, 0, 0]
    radius = diameter * 10000

    _, indices_visible = pcd_centered.hidden_point_removal(camera, radius)

    if visualize:
        pcd_visible = pcd_centered.select_by_index(indices_visible)
        m = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd_visible, m])

    return indices_visible


def color_pcd_by_two_groups(points, indices):
    colors = generate_random_colors(2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))

    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(
        np.zeros(np.asarray(pcd.points).shape)
    )

    points = pcd_colored.points

    for i in range(len(points)):
        pcd_colored.colors[i] = colors[0]

    for i in range(len(indices)):
        pcd_colored.colors[indices[i]] = colors[1]

    return pcd_colored


def color_pcd_by_labels(pcd, labels):
    """Cloud coloring after constructing an instance matrix by labels colors

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        cloud for coloring
    labels : numpy.array
        row in the instance matrix
    """

    colors = generate_random_colors(len(labels) + 1)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(
        np.zeros(np.asarray(pcd.points).shape)
    )

    for i in range(len(pcd_colored.points)):
        pcd_colored.colors[i] = colors[labels[i]]

    return pcd_colored


def color_pcd_by_clusters_and_voxels(pcd, trace, clusters):
    """Cloud coloring after segmentation by colors of labeled clusters

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        cloud for coloring
    trace : list of IntVectors
        the i-th element stores the indices of the points of the original cloud that formed the i-th voxel
    clusters : list of lists
        the i-th element stores the indices of the points of the voxel cloud that fell into the i-th cluster
    """

    random_colors = generate_random_colors(len(clusters) + 1)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(
        np.zeros(np.asarray(pcd.points).shape)
    )

    for i in range(len(clusters)):
        cluster = clusters[i]
        for voxel in cluster:
            src_points = trace[voxel]
            for point in src_points:
                pcd_colored.colors[point] = random_colors[i + 1]

    return pcd_colored
