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

from utils.image_utils import generate_random_colors


def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def get_subpcd(pcd, indices):
    subpcd = o3d.geometry.PointCloud()
    subpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    return subpcd


# Строим карту в системе координат L0 (я думаю K0)
def build_map_wc(dataset, cam_name, start_index, end_index):
    map_wc = o3d.geometry.PointCloud()

    for i in range(start_index, end_index):
        T = dataset.get_lidar_pose(i)
        map_wc += copy.deepcopy(dataset.get_point_cloud(i)).transform(T)

    return map_wc


def build_map_wc_triangle_mesh(dataset, cam_name, start_index, end_index, visualize=False):
    map_wc = o3d.geometry.PointCloud()

    geometries = []
    for i in range(start_index, end_index):
        T = dataset.get_lidar_pose(i)
        # Сдвигаем все облака в систему координат L0 (я думаю K0)
        map_wc += copy.deepcopy(dataset.get_point_cloud(i)).transform(T)

        m = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # Положение камеры в системе координат L0 (я думаю K0)
        T_cam = T @ np.linalg.inv(dataset.get_camera_extrinsics(cam_name))
        geometries.append(m.transform(T_cam))

    if visualize:
        # Визуализируем карту и расположение камеры относительно нее    
        o3d.visualization.draw_geometries([map_wc] + geometries)

    return map_wc, geometries


# pcd_centered -- облако в системе координат камеры
def get_visible_points(pcd_centered, visualize=False):
    diameter = np.linalg.norm(
        np.asarray(pcd_centered.get_max_bound()) - np.asarray(pcd_centered.get_min_bound())
    )
    camera = [0, 0, 0]
    radius = diameter * 100

    _, indices_visible = pcd_centered.hidden_point_removal(camera, radius)

    if visualize:
        pcd_visible = pcd_centered.select_by_index(indices_visible)
        m = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd_visible, m])

    return indices_visible


# center -- pose around what point to take sphere 
def get_close_point_indices(pcd, center, R):
    pcd_centered = copy.deepcopy(pcd).transform(np.linalg.inv(center))
    
    points = np.asarray(pcd_centered.points)
    dists = np.linalg.norm(points, axis=1)
    return np.where(dists < R)[0]


def color_pcd_by_labels(pcd, labels):
    colors = generate_random_colors(len(labels) + 1)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd.points).shape))

    for i in range(len(pcd_colored.points)):
        pcd_colored.colors[i] = colors[labels[i]]

    return pcd_colored


def color_pcd_by_clusters(pcd, clusters):
    random_colors = generate_random_colors(len(clusters) + 1)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd.points).shape))

    for i in range(len(pcd_colored.points)):
        index = find_point_in_clusters(pcd.points[i], clusters)
        pcd_colored.colors[i] = random_colors[index + 1 if index != 10000 else 0]

    return pcd_colored


def find_point_in_clusters(point, clusters):
    for itr in range(len(clusters)):
        cluster = clusters[itr]
        for itr2 in range(len(cluster)):
            if (cluster[itr2] == point).all():
                return itr
    return -1


def color_pcd_by_clusters_and_neighbors(pcd_src, clusters, neighbors):
    random_colors = generate_random_colors(len(clusters) + 1)
    pcd_colored = copy.deepcopy(pcd_src)
    pcd_colored.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd_src.points).shape))

    for i in range(len(pcd_colored.points)):
        index = find_point_in_clusters(pcd_src.points[i], clusters)

        if index == -1: # для точки не нашли кластер, значит она была выброшена при voxel_down => красим в цвет соседа
            index = find_point_in_clusters(neighbors[i], clusters)

        pcd_colored.colors[i] = random_colors[index + 1 if index != -1 else 0]

    return pcd_colored


def color_pcd_by_clusters_common(pcd_src, clusters, neighbors):
    counter1 = 0
    counter2 = 0
    random_colors = generate_random_colors(len(clusters) + 1)

    pcd_colored1 = copy.deepcopy(pcd_src)
    pcd_colored1.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd_src.points).shape))

    for i in range(len(pcd_colored1.points)):
        index = find_point_in_clusters(pcd_src.points[i], clusters)
        if index == -1:
            counter1 += 1
        pcd_colored1.colors[i] = random_colors[index + 1 if index != -1 else 0]

    
    pcd_colored2 = copy.deepcopy(pcd_src)
    pcd_colored2.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd_src.points).shape))

    for i in range(len(pcd_colored2.points)):
        index = find_point_in_clusters(pcd_src.points[i], clusters)

        if index == -1: # для точки не нашли кластер, значит она была выброшена при voxel_down => красим в цвет соседа
            index = find_point_in_clusters(neighbors[i], clusters)
            if index != -1:
                counter2 += 1
            # if index == -1:
            #     counter2 += 1
            #     print("strange i = {}".format(i))

        pcd_colored2.colors[i] = random_colors[index + 1 if index != -1 else 0]

    return pcd_colored1, pcd_colored2, counter1, counter2
