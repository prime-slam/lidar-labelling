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

import numpy as np
import open3d as o3d

from utils.image_utils import generate_random_colors


def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def get_point_map(cam_name, pcd_dataset, start_index, end_index):
    base_cloud_index = start_index

    pcd_combined = o3d.geometry.PointCloud()
    for current_cloud_index in range(start_index + 1, end_index):
        pcd_combined = paired_association(
            cam_name,
            pcd_dataset,
            base_cloud_index,
            current_cloud_index,
            pcd_combined
        )

    return pcd_combined


def paired_association(cam_name, pcd_dataset, target_cloud_index, src_cloud_index, pcd_combined):
    target_cloud = pcd_dataset.get_point_cloud(target_cloud_index)
    src_cloud = pcd_dataset.get_point_cloud(src_cloud_index)

    matrix_src_cloud_to_target = pcd_dataset.calculate_pcd_motion_matrix(cam_name, src_cloud_index, target_cloud_index)

    src_cloud.transform(matrix_src_cloud_to_target)

    if len(pcd_combined.points) == 0:
        pcd_combined += target_cloud

    pcd_combined += src_cloud

    return pcd_combined


def remove_hidden_points(pcd):
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    )
    camera = [0, 0, 0]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    return pcd.select_by_index(pt_map)


def visualize_by_clusters(clusters, pcd):
    random_colors = generate_random_colors(len(clusters) + 1)

    colors = []
    for i in range(len(np.asarray(pcd.points))):
        index = find_point_in_clusters(clusters, pcd.points[i])
        colors.append(random_colors[index + 1 if index != 1000 else 0])

    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)
    return pcd


def find_point_in_clusters(clusters, point):
    for itr in range(len(clusters)):
        cluster = clusters[itr]
        for itr2 in range(len(cluster)):
            if (cluster[itr2] == point).all():
                return itr
    return 1000


def find_points_in_sphere(src_points, R, enc_coo_non_zero_instances):
    x0, y0, z0 = find_center_of_sphere(src_points)

    points_in_sphere = []
    enc_coo_non_zero_instances_points_in_sphere = []
    for itr in range(len(src_points)):
        point = src_points[itr]
        dx = (point[0] - x0) ** 2
        dy = (point[1] - y0) ** 2
        dz = (point[2] - z0) ** 2

        if dx + dy + dz <= R ** 2:
            points_in_sphere.append(point)
            enc_coo_non_zero_instances_points_in_sphere.append(enc_coo_non_zero_instances[itr])

    return points_in_sphere, enc_coo_non_zero_instances_points_in_sphere


def find_center_of_sphere(points):
    x, y, z = 0, 0, 0
    for point in range(len(points)):
        x += points[point][0]
        y += points[point][1]
        z += points[point][2]

    x0 = x / len(points)
    y0 = y / len(points)
    z0 = z / len(points)
    print("central point = ({}, {}, {})".format(x0, y0, z0))
    return x0, y0, z0


def visualize_from_one_hot_encoding(encoding_matrix, enc, pcds, start_index, view_index):
    instances = enc.inverse_transform(encoding_matrix)
    print(instances[40])
    print(instances[40][view_index - start_index])
    print(len(instances))
    print(pcds[view_index - start_index])

    pcd = pcds[view_index - start_index]

    random_colors = generate_random_colors(500)

    colors = []

    for i in range(len(np.asarray(pcd.points))):
        colors.append(random_colors[int(instances[i][view_index - start_index])])

    print(len(colors))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)

    return pcd
