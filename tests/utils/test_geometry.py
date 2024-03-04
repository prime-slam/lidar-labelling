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

import pytest

from src.utils.geometry_utils import find_intersection
from src.utils.geometry_utils import find_union
from src.utils.geometry_utils import calculate_area


def test_find_intersection_none_result():
    bbox1 = [2, 1, 3, 2]
    bbox2 = [6, 4, 5, 4]
    expected_bbox_intersection = None

    actual_bbox_intersection = find_intersection(bbox1, bbox2)
    assert actual_bbox_intersection == expected_bbox_intersection


@pytest.mark.parametrize(
    "bbox1, " "bbox2, " "expected_bbox_intersection",
    [
        (
            [7, 1, 2, 6],
            [6, 4, 5, 4],
            [7, 4, 2, 3],
        )
    ],
)
def test_find_intersection(
    bbox1,
    bbox2,
    expected_bbox_intersection,
):
    actual_bbox_intersection = find_intersection(bbox1, bbox2)
    assert actual_bbox_intersection == expected_bbox_intersection


@pytest.mark.parametrize(
    "bbox1, " "bbox2, " "expected_bbox_union",
    [
        (
            [2, 1, 3, 2],
            [6, 4, 5, 4],
            [2, 1, 9, 7],
        )
    ],
)
def test_find_union(
    bbox1,
    bbox2,
    expected_bbox_union,
):
    actual_bbox_union = find_union(bbox1, bbox2)
    assert actual_bbox_union == expected_bbox_union


def test_calculate_area():
    bbox = [2, 1, 3, 2]
    expected_bbox_area = 6

    actual_bbox_area = calculate_area(bbox)
    assert actual_bbox_area == expected_bbox_area
