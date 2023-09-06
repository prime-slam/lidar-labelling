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

from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self, dataset):
        self._dataset = dataset
        self._cam_poses = self.dataset.poses
        super().__init__()

    @property
    def dataset(self):
        return self._dataset

    @property
    def cam_poses(self):
        return self._cam_poses

    @abstractmethod
    def get_point_cloud(self, index):
        pass

    @abstractmethod
    def get_camera_names(self):
        pass

    @abstractmethod
    def get_camera_image(self, cam_name, index):
        pass

    @abstractmethod
    def get_camera_intrinsics(self, cam_name):
        pass

    @abstractmethod
    def get_camera_extrinsics(self, cam_name):
        pass

    def get_lidar_poses(self, index, cam_name):
        return (
                np.linalg.inv(self.get_camera_extrinsics(cam_name))
                @ self.cam_poses[index]
                @ self.get_camera_extrinsics(cam_name)
        )

    def prepare_points_before_mapping(self, cam_name, pcd, start_pcd_index, image_index):
        pcd_L0 = self.transform_to_pcd0(pcd=pcd, cam_name=cam_name, start_pcd_index=start_pcd_index)
        pcd_Ki = self.transform_pcd0_to_cami_coordinate_system(pcd=pcd_L0, i=image_index)

        return pcd_Ki

    def transform_to_pcd0(self, pcd=None, cam_name="cam0", start_pcd_index=0):
        matrix = (
                np.linalg.inv(self.get_camera_extrinsics(cam_name))
                @ self.cam_poses[start_pcd_index]
                @ self.get_camera_extrinsics(cam_name)
        )

        return pcd.transform(matrix)

    def transform_pcd0_to_cami_coordinate_system(self, pcd=None, cam_name="cam0", i=0):
        matrix = np.linalg.inv(self.cam_poses[i]) @ self.get_camera_extrinsics(cam_name)

        return pcd.transform(matrix)

    def calculate_pcd_motion_matrix(self, cam_name, src_index, target_index):
        target_cloud_poses = self.cam_poses[target_index]
        src_cloud_poses = self.cam_poses[src_index]

        src_cam_to_target_poses = np.linalg.inv(target_cloud_poses) @ src_cloud_poses
        matrix_src_cloud_to_target = (
                np.linalg.inv(self.get_camera_extrinsics(cam_name))
                @ src_cam_to_target_poses
                @ self.get_camera_extrinsics(cam_name)
        )

        return matrix_src_cloud_to_target
