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

from scipy.spatial.distance import cdist

from pcd_dataset.kitti_dataset import KittiDataset

from services.normalized_cut_service import normalized_cut

from services.preprocessing.common.config import ConfigDTO
from services.preprocessing.init.map import InitMapProcessor
from services.preprocessing.init.instances_matrix import InitInstancesMatrixProcessor
from services.preprocessing.not_zero import SelectionNotZeroProcessor
from services.preprocessing.in_cube import SelectionInCubeProcessor
from services.preprocessing.statistical_outlier import StatisticalOutlierProcessor
from services.preprocessing.voxel_down import VoxelDownProcessor

from utils.distances_utils import remove_isolated_points
from utils.distances_utils import sam_label_distance
from utils.pcd_utils import color_pcd_by_clusters_and_voxels


def main():
    dataset_path = 'dataset/'
    sequence = '00'
    image_instances_path = 'pipeline/vfm-labels/sam/00/'
    kitti = KittiDataset(dataset_path, sequence, image_instances_path)

    config = ConfigDTO(**{
    'dataset': kitti,
    'start_index': 19,
    'end_index': 23,
    'start_image_index_offset': 3,
    'cam_name': 'cam2',
    'R': 12,
    'nb_neighbors': 30,
    'std_ratio': 5.0,
    'voxel_size': 0.25
    })

    init_pcd = InitMapProcessor().process(config)
    points2instances = InitInstancesMatrixProcessor().process(config, init_pcd)

    processors = [
        SelectionNotZeroProcessor(),
        SelectionInCubeProcessor(),
        StatisticalOutlierProcessor()
    ]

    pcd = copy.deepcopy(init_pcd)
    for processor in processors:
        pcd, points2instances = processor.process(config, pcd, points2instances)

    pcd_for_clustering = copy.deepcopy(pcd)

    pcd, points2instances, voxel_results = VoxelDownProcessor().process(config, pcd, points2instances)
    trace = voxel_results[2]

    points = np.asarray(pcd.points)
    spatial_distance = cdist(points, points)

    dist, masks = sam_label_distance(
        points2instances, spatial_distance, proximity_threshold=2, beta=10, alpha=2
    )

    dist, points, trace = remove_isolated_points(dist, points, trace)

    T = 0.2
    eigenval = 2
    clusters = normalized_cut(dist, np.array([i for i in range(len(points))], dtype=int), T, eigenval)

    pcd_clustered = color_pcd_by_clusters_and_voxels(pcd_for_clustering, trace, clusters)

    o3d.visualization.draw_geometries([pcd_clustered])


if __name__ == "__main__":
    main()
