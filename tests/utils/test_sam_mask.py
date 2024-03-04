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

from src.utils.sam_mask_utils import find_intersection_mask
from src.utils.sam_mask_utils import find_union_mask


@pytest.mark.parametrize(
    "mask1, " "mask2, " "expected_intersection_mask",
    [
        (
            {
                "segmentation": np.array(
                    [
                        [False, False, True, False, False],
                        [False, False, True, True, False],
                        [True, False, True, False, False],
                    ]
                ),
                "bbox": [0, 0, 4, 3],
                "area": 5,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, True],
                        [False, False, False, False, False],
                    ]
                ),
                "bbox": [2, 1, 3, 1],
                "area": 3,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, False],
                        [False, False, False, False, False],
                    ]
                ),
                "bbox": [2, 1, 2, 1],
                "area": 2,
            },
        )
    ],
)
def test_find_intersection_mask(
    mask1,
    mask2,
    expected_intersection_mask,
):
    actual_intersection_mask = find_intersection_mask(mask1, mask2)
    assert (
        actual_intersection_mask["segmentation"]
        == expected_intersection_mask["segmentation"]
    ).all()
    assert actual_intersection_mask["bbox"] == expected_intersection_mask["bbox"]
    assert actual_intersection_mask["area"] == expected_intersection_mask["area"]


@pytest.mark.parametrize(
    "mask1, " "mask2",
    [
        (
            {
                "segmentation": np.array(
                    [
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [False, False, False, False, False],
                    ]
                ),
                "bbox": [0, 0, 2, 2],
                "area": 4,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, True, True],
                    ]
                ),
                "bbox": [3, 2, 2, 1],
                "area": 2,
            },
        )
    ],
)
def test_find_intersection_mask_none_result(
    mask1,
    mask2,
):
    actual_intersection_mask = find_intersection_mask(mask1, mask2)
    assert actual_intersection_mask == None


@pytest.mark.parametrize(
    "mask1, " "mask2, " "expected_union_mask",
    [
        (
            {
                "segmentation": np.array(
                    [
                        [False, False, True, False, False],
                        [False, False, True, True, False],
                        [True, False, True, False, False],
                    ]
                ),
                "bbox": [0, 0, 4, 3],
                "area": 5,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, False, False, False],
                        [False, False, True, True, True],
                        [False, False, False, False, False],
                    ]
                ),
                "bbox": [2, 1, 3, 1],
                "area": 3,
            },
            {
                "segmentation": np.array(
                    [
                        [False, False, True, False, False],
                        [False, False, True, True, True],
                        [True, False, True, False, False],
                    ]
                ),
                "bbox": [0, 0, 5, 3],
                "area": 6,
            },
        )
    ],
)
def test_find_union_mask(
    mask1,
    mask2,
    expected_union_mask,
):
    actual_union_mask = find_union_mask(mask1, mask2)
    assert (
        actual_union_mask["segmentation"]
        == expected_union_mask["segmentation"]
    ).all()
    assert actual_union_mask["bbox"] == expected_union_mask["bbox"]
    assert actual_union_mask["area"] == expected_union_mask["area"]
