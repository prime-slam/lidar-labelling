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
import pickle

from scipy.spatial.distance import cdist

from src.datasets.kitti_dataset import KittiDataset

from src.services.distance.isolated import RemovingIsolatedPointsProcessor
from src.services.distance.connected_component import (
    ExtractionLargestConnectedComponentProcessor,
)
from src.services.normalized_cut_service import normalized_cut
from src.services.preprocessing.common.config import ConfigDTO
from src.services.preprocessing.init.map import InitMapProcessor
from src.services.preprocessing.init.instances_matrix import (
    InitInstancesMatrixProcessor,
)
from src.services.preprocessing.not_zero import SelectionNotZeroProcessor
from src.services.preprocessing.in_cube import SelectionInCubeProcessor
from src.services.preprocessing.statistical_outlier import StatisticalOutlierProcessor
from src.services.preprocessing.voxel_down import VoxelDownProcessor

from src.utils.distances_utils import sam_label_distance
from src.utils.gt_utils import build_sem_inst_label_arrays


dataset_path = "dataset/"
sequence = "00"
image_instances_path = "pipeline/vfm-labelss/sam/00/"
gt_labels_path = "dataset/sequences/00/labels/"
kitti = KittiDataset(dataset_path, sequence, image_instances_path)


def build_tuple_bin_saving(
    config,
    pcd_for_clustering,
    trace,
    clusters,
    inst_label_array_for_clustering,
    sem_label_array_for_clustering,
):
    params = {}
    params["alpha_physical_distance"] = config.alpha_physical_distance
    params["beta_instance_distance"] = config.beta_instance_distance
    params["T_normalized_cut"] = config.T_normalized_cut
    params["reduce_detail_int_to_union_threshold"] = (
        config.reduce_detail_int_to_union_threshold
    )
    params["reduce_detail_int_to_mask_threshold"] = (
        config.reduce_detail_int_to_mask_threshold
    )

    trace_arrays = []
    for int_vector in trace:
        trace_arrays.append(np.asarray(int_vector))

    return (
        params,
        np.asarray(pcd_for_clustering.points),
        trace_arrays,
        clusters,
        inst_label_array_for_clustering,
        sem_label_array_for_clustering,
    )


def segment_pcds(config):
    init_pcd = InitMapProcessor().process(config)
    points2instances = InitInstancesMatrixProcessor().process(config, init_pcd)

    sem_label_array_src, inst_label_array_src = build_sem_inst_label_arrays(
        gt_labels_path, config.start_index, config.end_index
    )

    processors = [
        SelectionNotZeroProcessor(),
        SelectionInCubeProcessor(),
        StatisticalOutlierProcessor(),
    ]

    pcd = copy.deepcopy(init_pcd)
    for processor in processors:
        pcd, points2instances, indices = processor.process(
            config, pcd, points2instances
        )
        inst_label_array_src = inst_label_array_src[indices]
        sem_label_array_src = sem_label_array_src[indices]

    pcd_for_clustering = copy.deepcopy(pcd)
    points2instances_pcd_for_clustering = copy.deepcopy(points2instances)
    inst_label_array_for_clustering = copy.deepcopy(inst_label_array_src)
    sem_label_array_for_clustering = copy.deepcopy(sem_label_array_src)

    pcd, points2instances, trace = VoxelDownProcessor().process(
        config, pcd, points2instances
    )

    points = np.asarray(pcd.points)
    spatial_distance = cdist(points, points)

    dist, masks = sam_label_distance(
        points2instances,
        spatial_distance,
        3,
        config.beta_instance_distance,
        config.alpha_physical_distance,
    )

    distance_processors = [
        RemovingIsolatedPointsProcessor(),
        ExtractionLargestConnectedComponentProcessor(),
    ]

    for processor in distance_processors:
        dist, points, trace = processor.process(dist, points, trace)

    eigenval = 2
    clusters = normalized_cut(
        dist,
        np.array([i for i in range(len(points))], dtype=int),
        config.T_normalized_cut,
        eigenval,
    )

    return build_tuple_bin_saving(
        config,
        copy.deepcopy(pcd_for_clustering),
        copy.deepcopy(trace),
        copy.deepcopy(clusters),
        copy.deepcopy(inst_label_array_for_clustering),
        copy.deepcopy(sem_label_array_for_clustering),
    )


def process_kitti(
    from_num, to_num, id_exec, alpha_physical_distance, beta_instance_distance
):
    T_normalized_cut = 0.03

    reduce_detail_int_to_union_threshold = 0.5
    reduce_detail_int_to_mask_threshold = 0.6

    current_from_num = from_num

    while current_from_num < to_num:
        start_index = current_from_num
        end_index = start_index + 4
        config = ConfigDTO(
            **{
                "dataset": kitti,
                "start_index": start_index,
                "end_index": end_index,
                "start_image_index_offset": 0,
                "alpha_physical_distance": alpha_physical_distance,
                "beta_instance_distance": beta_instance_distance,
                "T_normalized_cut": T_normalized_cut,
                "reduce_detail_int_to_union_threshold": reduce_detail_int_to_union_threshold,
                "reduce_detail_int_to_mask_threshold": reduce_detail_int_to_mask_threshold,
                "cam_name": "cam2",
                "R": 18,
                "nb_neighbors": 25,
                "std_ratio": 5.0,
                "voxel_size": 0.25,
            }
        )

        result_tuple = segment_pcds(config)

        file_name = (
            "experiment_bin_0704_{}_sem_offset0_T0l03/start{}_end{}.pickle".format(
                id_exec, config.start_index, config.end_index
            )
        )
        new_file = open(file_name, "w")
        new_file.close()

        with open(file_name, "wb") as file:
            pickle.dump(result_tuple, file)

        print("start_index={}, end_index={} done".format(start_index, end_index))
        current_from_num = end_index


def main():
    alpha_physical_distance_4 = 5
    beta_instance_distance_4 = 5
    process_kitti(0, 4540, 4, alpha_physical_distance_4, beta_instance_distance_4)


if __name__ == "__main__":
    main()
