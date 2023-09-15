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

import cv2
import numpy as np
import open3d as o3d
import pykitti

from pathlib import Path

from pcd_dataset.abstract_pcd_dataset import AbstractDataset


class KittiDataset(AbstractDataset):
    def __init__(self, dataset_path, sequence, image_instances_path):
        super().__init__(pykitti.odometry(dataset_path, sequence))
        self.image_instances_path = image_instances_path

    def get_point_cloud(self, index):
        points = self.dataset.get_velo(index)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def get_camera_names(self):
        return ['cam0', 'cam1', 'cam2', 'cam3']

    def get_camera_image(self, cam_name, index):
        image, color = None, None
        if cam_name == 'cam0':
            image, color = self.dataset.get_cam0(index), cv2.COLOR_GRAY2BGR
        elif cam_name == 'cam1':
            image, color = self.dataset.get_cam1(index), cv2.COLOR_GRAY2BGR
        elif cam_name == 'cam2':
            image, color = self.dataset.get_cam2(index), cv2.COLOR_RGB2BGR
        elif cam_name == 'cam3':
            image, color = self.dataset.get_cam3(index), cv2.COLOR_RGB2BGR
        return cv2.cvtColor(np.array(image), color)

    def get_image_instances(self, cam_name, index):
        masks_path = Path.cwd().joinpath(self.image_instances_path, cam_name, '{}.npz'.format(str(index).zfill(6)))
        return np.load(masks_path, allow_pickle=True)['masks']

    def get_camera_intrinsics(self, cam_name):
        if cam_name == 'cam0':
            return self.dataset.calib.K_cam0
        elif cam_name == 'cam1':
            return self.dataset.calib.K_cam1
        elif cam_name == 'cam2':
            return self.dataset.calib.K_cam2
        elif cam_name == 'cam3':
            return self.dataset.calib.K_cam3
        else:
            return None

    def get_camera_extrinsics(self, cam_name):
        if cam_name == 'cam0':
            return self.dataset.calib.T_cam0_velo
        elif cam_name == 'cam1':
            return self.dataset.calib.T_cam1_velo
        elif cam_name == 'cam2':
            return self.dataset.calib.T_cam2_velo
        elif cam_name == 'cam3':
            return self.dataset.calib.T_cam3_velo
        else:
            return None
