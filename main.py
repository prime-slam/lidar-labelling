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
from sklearn.preprocessing import OneHotEncoder

from awesome_normalized_cut import normalized_cut
from mapping.simple_mapping import SimpleMapping
from mapping.sparse_matrix_utils import construct_coo_matrix_for_multiple_views
from pcd_dataset.kitti_dataset import KittiDataset
from utils.pcd_utils import visualize_by_clusters, find_points_in_sphere
from utils.distances_utils import calculate_distances, print_pairwise_distances


def main():
    dataset_path = 'dataset/'
    sequence = '00'
    sam_labels_path = Path.cwd().joinpath('pipeline', 'vfm-labels', 'sam', sequence)
    kitti = KittiDataset(dataset_path, sequence, sam_labels_path)
    sm = SimpleMapping(kitti)

    coo_matrix, pcds = sm.get_combined_labeled_point_clouds('cam2', 20, 23)

    # whole map
    coo_result = construct_coo_matrix_for_multiple_views(coo_matrix, 20)

    # enc whole map
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    enc.fit(coo_result.T.toarray())
    enc_coo_result = enc.transform(coo_result.T.toarray()).toarray()
    inverse_enc_coo_result = enc.inverse_transform(enc_coo_result)

    # delete points that are unlabeled on all the views
    points, map_old_new_point_ind, enc_coo_non_zero_instances = get_non_zero_instances_points(
        pcds, enc_coo_result, inverse_enc_coo_result
    )

    # selection of points that are inside the sphere
    points_in_sphere, enc_coo_non_zero_instances_points_in_sphere = find_points_in_sphere(points, 4, enc_coo_non_zero_instances)
    pcd_baby = o3d.geometry.PointCloud()
    pcd_baby.points = o3d.utility.Vector3dVector(points_in_sphere)

    distances_points_in_sphere = calculate_distances(enc.inverse_transform(enc_coo_non_zero_instances_points_in_sphere))

    clusters = normalized_cut(distances_points_in_sphere, np.asarray(points_in_sphere), 1.0001)

    print("number of clusters = {}".format(len(clusters)))
    print("number of points = {}".format(len(pcd_baby.points)))

    # pcd_baby_color = visualize_by_clusters(clusters, pcd_baby)
    # o3d.io.write_point_cloud('cloud_20_23_4m_T1-00018.pcd', pcd_baby_color)
    # o3d.visualization.draw_geometries([pcd_baby_color])

    # inverse_in_sphere = enc.inverse_transform(enc_coo_non_zero_instances_points_in_sphere)
    # for itr in range(len(clusters)):
    #     print(clusters[itr])
    #     if itr == 0:
    #         print("distance between {} and {} = {}".format(
    #             clusters[itr][0], clusters[itr][1], distances_points_in_sphere[clusters[itr][0], clusters[itr][1]]
    #         ))
    #         print("instances: point{} = {}".format(
    #             clusters[itr][0], inverse_in_sphere[clusters[itr][0]]
    #         ))
    #         print("instances: point{} = {}".format(
    #             clusters[itr][1], inverse_in_sphere[clusters[itr][1]]
    #         ))


def main1():
    x = 7
    # downpcd_trace = pcd_baby.voxel_down_sample_and_trace(0.5, pcd_baby.get_min_bound(), pcd_baby.get_max_bound(), True)
    #
    # # delete points that aren't in the downpcd
    # enc_coo_non_zero_instances_downpcd = get_enc_coo_non_zero_instances_downpcd(downpcd_trace,
    #                                                                             enc_coo_non_zero_instances,
    #                                                                             enc)
    # # instances of downpcd points in format: [ 2., 0., 16.]
    # inverse_enc_coo_non_zero_instances_downpcd = enc.inverse_transform(enc_coo_non_zero_instances_downpcd)
    #
    # distances_downpcd = calculate_distances(inverse_enc_coo_non_zero_instances_downpcd)
    #
    # print(len(enc_coo_non_zero_instances))
    #
    # clusters = normalized_cut(distances_downpcd, np.asarray(downpcd_trace[0].points), 1.0038)


def get_non_zero_instances_points(pcds, enc_coo_result, inverse_enc_coo_result):
    summary_pcd_points = []
    for itr in range(len(pcds)):
        summary_pcd_points += pcds[itr].points

    points = []
    map_old_new_point_ind = {}
    enc_coo_non_zero_instances = []
    for itr in range(len(inverse_enc_coo_result)):
        if (inverse_enc_coo_result[itr] != 0.0).any():
            points.append(summary_pcd_points[itr])
            pair = {itr: (len(points) - 1)}
            map_old_new_point_ind.update(pair)
            enc_coo_non_zero_instances.append(enc_coo_result[itr])

    return points, map_old_new_point_ind, enc_coo_non_zero_instances


def print_instances(enc_coo, enc, start, end):
    inverse_transform = enc.inverse_transform(enc_coo)
    for itr in range(start, end):
        print("point{} = {}".format(itr, inverse_transform[itr]))


def get_enc_coo_non_zero_instances_downpcd(downpcd_trace, enc_coo_non_zero_instances, enc):
    enc_coo_non_zero_instances_downpcd = []
    inverse_transform = enc.inverse_transform(enc_coo_non_zero_instances)
    voxel_down_sample_points = downpcd_trace[2]
    for point in range(len(voxel_down_sample_points)):
        legend = []
        for itr in range(len(voxel_down_sample_points[point])):
            if itr == 0:
                legend = inverse_transform[voxel_down_sample_points[point][itr]]
                enc_coo_non_zero_instances_downpcd.append(
                    enc_coo_non_zero_instances[voxel_down_sample_points[point][itr]])
                print("legend for voxel_down_sample_points[point] = {} is {}".format(voxel_down_sample_points[point],
                                                                                     legend))
            else:
                if (inverse_transform[voxel_down_sample_points[point][itr]] != legend).any():
                    print("instances didn't match the legend")
    return enc_coo_non_zero_instances_downpcd


if __name__ == "__main__":
    main()
