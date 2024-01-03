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

from services.preprocessing.voxel_down import VoxelDownProcessor

from tests.test_data import config


@pytest.mark.parametrize(
    "src_points, "
    "src_points2instances, "
    "expected_points, "
    "expected_points2instances, "
    "expected_trace",
    [
        (
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.11, 0.11, 0.11],
                    [0.111, 0.111, 0.111],
                    [-10000, -20000, -30000],
                    [1000, 2000, 3000],
                    [1000.1, 2000.2, 3000.3],
                ]
            ),
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 0, 0],
                    [1, 2, 3, 4, 5, 6],
                    [18, 17, 56, 78, 90, 23],
                    [1, 2, 3, 0, 0, 6],
                    [1, 2, 3, 0, 0, 6],
                ]
            ),
            np.array(
                [
                    [0.107, 0.107, 0.107],
                    [-10000, -20000, -30000],
                    [1000.05, 2000.1, 3000.15],
                ]
            ),
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [18, 17, 56, 78, 90, 23],
                    [1, 2, 3, 0, 0, 6],
                ]
            ),
            [
                o3d.utility.IntVector([0, 1, 2]),
                o3d.utility.IntVector([3]),
                o3d.utility.IntVector([4, 5]),
            ],
        )
    ],
)
def test_voxel_down_sample(
    src_points, src_points2instances, expected_points, expected_points2instances, expected_trace
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_points)

    actual_pcd, actual_points2instances, actual_trace = VoxelDownProcessor().process(
        config, pcd, src_points2instances
    )

    assert check_result_points(actual_pcd, expected_points) == len(expected_points)
    assert check_result_points2instances(actual_points2instances, expected_points2instances) == expected_points2instances.shape[0]
    assert check_result_trace(actual_trace, expected_trace) == len(expected_trace)


def check_result_points(actual_pcd, expected_points):
    actual_points = np.asarray(actual_pcd.points)

    number_of_matched_positions = 0
    for point1 in actual_points:
        for point2 in expected_points:
            if (point1 == point2).all():
                number_of_matched_positions += 1
                break

    return number_of_matched_positions


def check_result_points2instances(actual_points2instances, expected_points2instances):
    matched_rows_in_expected = []

    number_of_matched_rows = 0
    for row1 in actual_points2instances:
        for id, row2 in enumerate(expected_points2instances):
            if id not in matched_rows_in_expected:
                if (row1 == row2).all():
                    number_of_matched_rows += 1
                    matched_rows_in_expected.append(id)
                    break

    return number_of_matched_rows


def check_result_trace(actual_trace, expected_trace):
    matched_rows_in_expected = []

    number_of_matched_rows = 0
    for row1 in actual_trace:
        int_vector1 = np.asarray(row1)

        for id, row2 in enumerate(expected_trace):
            int_vector2 = np.asarray(row2)
            if len(int_vector1) != len(int_vector2):
                continue

            if id not in matched_rows_in_expected:
                if (int_vector1 == int_vector2).all():
                    number_of_matched_rows += 1
                    matched_rows_in_expected.append(id)
                    break

    return number_of_matched_rows
