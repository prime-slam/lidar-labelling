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

import numpy as np
import open3d as o3d
import pickle

from sklearn.cluster import HDBSCAN


def segment_pcds_by_hdbscan(start_index, end_index):

    file_name = "experiment_bin/start{}_end{}.pickle".format(start_index, end_index)

    with open(file_name, "rb") as file:
        data = pickle.load(file)  # prepared dense cloud

    pcd_hdbscan_voxel_down_points = np.asarray(data[2]["voxel_pcd_original_points"])
    pcd_hdbscan_voxel_down = o3d.geometry.PointCloud()
    pcd_hdbscan_voxel_down.points = o3d.utility.Vector3dVector(
        pcd_hdbscan_voxel_down_points
    )

    clusterer = HDBSCAN()
    clusters = clusterer.fit_predict(np.asarray(pcd_hdbscan_voxel_down.points))

    return (
        {
            "hdbscan_clustered_voxel_pcd_original_points": np.asarray(
                pcd_hdbscan_voxel_down.points
            )
        },
        {"hdbscan_voxel_trace_original": data[3]["voxel_trace_original"]},
        {"hdbscan_clusters": clusters},
        {"inst_label_array_for_clustering": data[6]["inst_label_array_for_clustering"]},
    )


def process_kitti_hdbscan(from_num, to_num):

    current_from_num = from_num
    step = 4

    while current_from_num < to_num:
        start_index = current_from_num
        end_index = start_index + step

        result_tuple = segment_pcds_by_hdbscan(start_index, end_index)

        file_name = "experiment_bin_hdbscan/start{}_end{}.pickle".format(
            start_index, end_index
        )  # hdbscan results
        new_file = open(file_name, "w")
        new_file.close()

        with open(file_name, "wb") as file:
            pickle.dump(result_tuple, file)

        print("start_index={}, end_index={} done".format(start_index, end_index))
        current_from_num = end_index


def main():
    process_kitti_hdbscan(from_num=0, to_num=4540)


if __name__ == "__main__":
    main()
