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

from pathlib import Path
from scipy.spatial.distance import cdist

from awesome_normalized_cut import normalized_cut
from mapping.simple_mapping import SimpleMapping
from mapping.sparse_matrix_utils import construct_coo_matrix_for_multiple_views
from pcd_dataset.kitti_dataset import KittiDataset
from utils.pcd_utils import visualize_by_clusters, find_points_in_sphere
from utils.distances_utils import sam_label_distance


def main():
    dataset_path = 'dataset/'
    sequence = '00'
    sam_labels_path = Path.cwd().joinpath('pipeline', 'vfm-labels', 'sam', sequence)
    kitti = KittiDataset(dataset_path, sequence, sam_labels_path)
    sm = SimpleMapping(kitti)

    start_index = 3850
    end_index = 3853
    R = 5
    image_num = 1
    T = 0.98

    coo_matrix, pcds = sm.get_combined_labeled_point_clouds('cam2', start_index, end_index)

    # whole map
    coo_result = construct_coo_matrix_for_multiple_views(coo_matrix, start_index)

    # delete points that are unlabeled on all the views
    points, map_old_new_point_ind, non_zero_instances = get_non_zero_instances_points(pcds, coo_result)

    # selection of points that are inside the sphere
    points_in_sphere, instances_points_in_sphere = find_points_in_sphere(points, R, non_zero_instances)
    print("points_in_sphere = {}".format(len(points_in_sphere)))
    print("instances_points_in_sphere = {}".format(len(instances_points_in_sphere)))

    spatial_distance = cdist(points_in_sphere, points_in_sphere)
    dist, masks = sam_label_distance(instances_points_in_sphere[:, :image_num], spatial_distance, 2, 10)

    clusters = normalized_cut(dist, np.asarray(points_in_sphere), T)

    pcd_baby = o3d.geometry.PointCloud()
    pcd_baby.points = o3d.utility.Vector3dVector(points_in_sphere)

    print("number of clusters = {}".format(len(clusters)))
    print("number of points = {}".format(len(pcd_baby.points)))

    pcd_baby_color = visualize_by_clusters(clusters, pcd_baby)

    filename = "cloud_{}_{}_{}image_{}m_T{}.pcd".format(start_index, end_index, image_num, R, "{}".format(T).replace('.', '-'))
    o3d.io.write_point_cloud(filename, pcd_baby_color)
    o3d.visualization.draw_geometries([pcd_baby_color])


def get_non_zero_instances_points(pcds, coo_result):
    summary_pcd_points = []
    for itr in range(len(pcds)):
        summary_pcd_points += pcds[itr].points

    array_point_instance = coo_result.T.todense()

    points = []
    map_old_new_point_ind = {}
    coo_non_zero_instances = []
    for itr in range(array_point_instance.shape[0]):
        if (array_point_instance[itr] != 0.0).any():
            points.append(summary_pcd_points[itr])
            pair = {itr: (len(points) - 1)}
            map_old_new_point_ind.update(pair)
            coo_non_zero_instances.append(array_point_instance[itr])

    return points, map_old_new_point_ind, coo_non_zero_instances


def print_instances(instances, start, end):
    for itr in range(start, end):
        print("point{} = {}".format(itr, instances[itr]))


# def get_enc_coo_non_zero_instances_downpcd(downpcd_trace, enc_coo_non_zero_instances, enc):
#     enc_coo_non_zero_instances_downpcd = []
#     inverse_transform = enc.inverse_transform(enc_coo_non_zero_instances)
#     voxel_down_sample_points = downpcd_trace[2]
#     for point in range(len(voxel_down_sample_points)):
#         legend = []
#         for itr in range(len(voxel_down_sample_points[point])):
#             if itr == 0:
#                 legend = inverse_transform[voxel_down_sample_points[point][itr]]
#                 enc_coo_non_zero_instances_downpcd.append(
#                     enc_coo_non_zero_instances[voxel_down_sample_points[point][itr]])
#                 print("legend for voxel_down_sample_points[point] = {} is {}".format(voxel_down_sample_points[point],
#                                                                                      legend))
#             else:
#                 if (inverse_transform[voxel_down_sample_points[point][itr]] != legend).any():
#                     print("instances didn't match the legend")
#     return enc_coo_non_zero_instances_downpcd


if __name__ == "__main__":
    main()
