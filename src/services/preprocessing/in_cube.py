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
import zope.interface

from src.services.preprocessing.common.interface import IProcessor
from src.utils.pcd_utils import get_subpcd


@zope.interface.implementer(IProcessor)
class SelectionInCubeProcessor:
    def process(self, config, pcd, points2instances):
        """Selection of points inside a cube with a given side.
        The cube starts from the lidar location at the moment of recording the config.start_index cloud.
        """

        T_first_cam = config.dataset.get_lidar_pose(config.start_index) @ np.linalg.inv(
            config.dataset.get_camera_extrinsics(config.cam_name)
        )

        close_point_indices = self.get_close_point_indices_cube(
            pcd, T_first_cam, config.R
        )

        pcd_in_cube = get_subpcd(pcd, close_point_indices)
        points2instances_in_cube = points2instances[close_point_indices]

        return pcd_in_cube, points2instances_in_cube

    def get_close_point_indices_cube(self, pcd, center, R):
        """Sampling pcd indices inside a cube

        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            original dense cloud
        center : matrix
            pose around what point to take cube
        R : int
            side of a cube
        """

        pcd_centered = copy.deepcopy(pcd).transform(np.linalg.inv(center))
        points = np.asarray(pcd_centered.points)

        vectors = np.array([[0, 0, point[2]] for point in points])
        dists = np.linalg.norm(vectors, axis=1)

        return np.where(dists < R)[0]
