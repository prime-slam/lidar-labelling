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


@zope.interface.implementer(IProcessor)
class RemovingIsolatedPointsProcessor:
    def process(self, distance_matrix, points, trace):
        """Removing isolated points that have all 0s in the distance matrix except the diagonal element"""

        mask_isolated = np.all(distance_matrix - np.eye(distance_matrix.shape[0]) == 0, axis=1)
        isolated_points = np.array([i for i in range(len(points))], dtype=int)[
            mask_isolated
        ]

        trace_copy = copy.deepcopy(trace)
        for index in sorted(isolated_points, reverse=True):
            del trace_copy[index]

        mask_not_isolated = np.any(distance_matrix - np.eye(distance_matrix.shape[0]) != 0, axis=1)

        return (
            distance_matrix[mask_not_isolated][:, mask_not_isolated],
            points[mask_not_isolated],
            trace_copy,
        )
