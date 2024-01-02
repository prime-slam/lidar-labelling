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
import open3d as o3d
import zope.interface

from services.preprocessing.common.interface import IProcessor


@zope.interface.implementer(IProcessor)
class InitMapProcessor:
    def process(self, config, pcd=None, points2instances=None):
        """Combining clouds from config.start_index to config.end_index into a dense map in the world coordinate system"""

        map_wc = o3d.geometry.PointCloud()

        for i in range(config.start_index, config.end_index):
            T = config.dataset.get_lidar_pose(i)
            map_wc += copy.deepcopy(config.dataset.get_point_cloud(i)).transform(T)

        return map_wc
