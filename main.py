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

from scipy.spatial.distance import cdist

from pcd_dataset.kitti_dataset import KittiDataset
from services.label_service import get_map_not_zero_in_sphere
from services.normalized_cut_service import normalized_cut
from utils.distances_utils import sam_label_distance
from utils.pcd_utils import color_pcd_by_clusters


def main():
    dataset_path = 'dataset_baby/'
    sequence = '00'
    image_instances_path = 'pipeline_baby/vfm-labels/sam/00/'
    kitti = KittiDataset(dataset_path, sequence, image_instances_path)

    start_index = 20
    end_index = 23
    cam_name = 'cam2'

    map, points2instances = get_map_not_zero_in_sphere(kitti, cam_name, start_index, end_index, 8, True)

    points = np.asarray(map.points)
    spatial_distance = cdist(points, points)
    dist, masks = sam_label_distance(points2instances, spatial_distance, 2, 10)

    T = 0.3
    clusters = normalized_cut(dist, np.asarray(points), T)

    print("len(clusters) = {}".format(len(clusters)))

    pcd_clustered = color_pcd_by_clusters(map, clusters)
    o3d.visualization.draw_geometries([pcd_clustered])


if __name__ == "__main__":
    main()
