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

import numpy as np
import pytest
import open3d as o3d

from src.utils.distances_utils import remove_isolated_points


@pytest.mark.parametrize(
    "dist, "
    "points, "
    "trace, "
    "expected_dist, "
    "expected_points, "
    "expected_trace",
    [
        (
            np.array(
                [
                    [1.0, 0.0038681, 0.0004307, 0.0],
                    [0.0811789, 1.0, 0.0817001, 0.0],
                    [0.0004307, 0.1346603, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [63.79, 11.29, 2.37],
                    [64.8, 11.45, 2.8],
                    [65.13, 11.45, 3.14],
                    [21.78, -0.346, -1.52],
                ]
            ),
            [
                o3d.utility.IntVector([1000, 7891, 452]),
                o3d.utility.IntVector([33]),
                o3d.utility.IntVector([224, 4565]),
                o3d.utility.IntVector([14, 8905]),
            ],
            np.array(
                [
                    [1.0, 0.0038681, 0.0004307],
                    [0.0811789, 1.0, 0.0817001],
                    [0.0004307, 0.1346603, 1.0],
                ]
            ),
            np.array(
                [
                    [63.79, 11.29, 2.37],
                    [64.8, 11.45, 2.8],
                    [65.13, 11.45, 3.14],
                ]
            ),
            [
                o3d.utility.IntVector([1000, 7891, 452]),
                o3d.utility.IntVector([33]),
                o3d.utility.IntVector([224, 4565]),
            ],
        )
    ],
)
def test_remove_isolated_points(
    dist,
    points,
    trace,
    expected_dist,
    expected_points,
    expected_trace,
):
    actual_dist, actual_points, actual_trace = remove_isolated_points(
        dist, points, trace
    )

    assert (actual_dist == expected_dist).all()
    assert (actual_points == expected_points).all()
    assert actual_trace == expected_trace
