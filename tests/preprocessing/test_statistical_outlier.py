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
import open3d as o3d
import pytest

from src.services.preprocessing.statistical_outlier import StatisticalOutlierProcessor

from tests.test_data import config


@pytest.mark.parametrize(
    "src_points, "
    "src_points2instances, "
    "expected_points, "
    "expected_points2instances",
    [
        (
            np.array(
                [
                    [0.1, 0.1, 2],
                    [0.2, 0.2, 3],
                    [0.3, 0.3, 4],
                    [0.4, 0.4, 2],
                    [0.5, 0.5, 6],
                    [-100000, -300000, -999999],
                    [0.7, 0.7, 5],
                    [0.8, 0.8, 3],
                    [0.9, 0.9, 1],
                ]
            ),
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [1, 0, 3, 4, 5, 6],
                    [1, 2, 0, 4, 5, 6],
                    [1, 2, 3, 0, 5, 6],
                    [1, 2, 3, 4, 0, 6],
                    [1, 2, 3, 4, 5, 0],
                    [0, 0, 3, 4, 5, 6],
                    [1, 0, 0, 4, 5, 6],
                ]
            ),
            np.array(
                [
                    [0.1, 0.1, 2],
                    [0.2, 0.2, 3],
                    [0.3, 0.3, 4],
                    [0.4, 0.4, 2],
                    [0.5, 0.5, 6],
                    [0.7, 0.7, 5],
                    [0.8, 0.8, 3],
                    [0.9, 0.9, 1],
                ]
            ),
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [1, 0, 3, 4, 5, 6],
                    [1, 2, 0, 4, 5, 6],
                    [1, 2, 3, 0, 5, 6],
                    [1, 2, 3, 4, 5, 0],
                    [0, 0, 3, 4, 5, 6],
                    [1, 0, 0, 4, 5, 6],
                ]
            ),
        )
    ],
)
def test_remove_statistical_outlier(
    src_points, src_points2instances, expected_points, expected_points2instances
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_points)

    actual_pcd, actual_points2instances, indices = (
        StatisticalOutlierProcessor().process(config, pcd, src_points2instances)
    )

    assert (np.asarray(actual_pcd.points) == expected_points).all()
    assert (actual_points2instances == expected_points2instances).all()
