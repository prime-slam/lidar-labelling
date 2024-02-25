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

from src.services.distance.interface import IProcessor
from src.utils.distances_utils import dfs
from src.utils.pcd_utils import color_pcd_by_two_groups
from src.utils.pcd_utils import visualize_pcd


@zope.interface.implementer(IProcessor)
class ExtractionLargestConnectedComponentProcessor:
    def process(self, distance_matrix, points, trace):
        """Extraction the largest connected component of a graph using the dfs algorithm"""

        num_vertices = len(points)
        half_num_vertices = num_vertices // 2
        visited_vertices = np.array([False for i in range(num_vertices)], dtype=bool)
        for i in range(num_vertices):
            visited_vertices = dfs(distance_matrix, i)
            if visited_vertices.sum() >= half_num_vertices:
                break

        not_visited_vertices = [
            vertex
            for vertex, is_visited in enumerate(visited_vertices)
            if not is_visited
        ]

        # Visualization of the extracted connectivity component against the background of the entire cloud
        visualize_pcd(color_pcd_by_two_groups(points, not_visited_vertices))

        trace_copy = copy.deepcopy(trace)
        for index in sorted(not_visited_vertices, reverse=True):
            del trace_copy[index]

        return (
            distance_matrix[visited_vertices][:, visited_vertices],
            points[visited_vertices],
            trace_copy,
        )
