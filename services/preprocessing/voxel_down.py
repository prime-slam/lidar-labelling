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

import copy
import numpy as np
import numpy_indexed as npi
import zope.interface

from services.preprocessing.common.interface import IProcessor


@zope.interface.implementer(IProcessor)
class VoxelDownProcessor:

    def process(self, config, pcd, points2instances):
        return self.build_o3d_voxel_pcd(
            pcd, points2instances, config.start_index, config.end_index, config.start_image_index_offset, config.voxel_size
        )


    def build_o3d_voxel_pcd(self, pcd, points2instances, start_index, end_index, image_offset, voxel_size):
        pcd_copy = copy.deepcopy(pcd)

        min_bound = pcd_copy.get_min_bound()
        max_bound = pcd_copy.get_max_bound()

        downpcd_trace = pcd_copy.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, True)

        downpcd = downpcd_trace[0]
        list_int_vectors = downpcd_trace[2]

        image_count = end_index - start_index + image_offset
        upd_points2instances = np.zeros((len(list_int_vectors), image_count), dtype=int)

        for i in range(len(list_int_vectors)):
            int_vector_array = np.asarray(list_int_vectors[i])
            instances = []
            for j in range(len(int_vector_array)):
                instances.append(points2instances[int_vector_array[j]])
            instances_array = np.asarray(instances)
            
            voxel_instance = npi.mode(instances_array)

            upd_points2instances[i] = voxel_instance

        return downpcd, upd_points2instances, downpcd_trace
