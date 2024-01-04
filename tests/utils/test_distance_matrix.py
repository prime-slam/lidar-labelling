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

from src.utils.distances_utils import sam_label_distance


@pytest.mark.parametrize(
    "sam_features, "
    "spatial_distance, "
    "expected_label_distance_beta5_alpha001, "
    "expected_label_distance_beta001_alpha5",
    [
        (
            np.array(
                [
                    [1, 2, 3, 4],
                    [1, 2, 5, 6],
                    [1, 2, 3, 4],
                    [0, 17, 13, 29],
                ]
            ),
            np.array(
                [
                    [0.   ,  1.11,  1.55, 43.76],
                    [1.11 ,  0.  ,  0.47, 44.81],
                    [1.55 ,  0.47,  0.  , 45.16],
                    [43.76, 44.81, 45.16,  0.  ],
                ]
            ),
            np.array(
                [
                    [1.       , 0.0811789, 0.9851119, 0.],
                    [0.0811789, 1.       , 0.0817001, 0.],
                    [0.9851119, 0.0817001, 1.       , 0.],
                    [0.       , 0.       , 0.       , 1.],
                ]
            ),
            np.array(
                [
                    [1.       , 0.0038681, 0.0004307, 0.],
                    [0.0038681, 1.       , 0.1346603, 0.],
                    [0.0004307, 0.1346603, 1.       , 0.],
                    [0.       , 0.       , 0.       , 1.],
                ]
            ),
        )
    ],
)
def test_label_distance_calculation(
    sam_features,
    spatial_distance,
    expected_label_distance_beta5_alpha001,
    expected_label_distance_beta001_alpha5,
):
    """The alpha and beta parameters indicate how significant physical distance (alpha)
    and instance similarity (beta) when calculating distance.
    """

    epsilon = 0.000001

    # 1) in this case, the most important criterion is proximity in instances
    #    instances of points 0 and 2 completely coincide, so they have a high distance value (0.985),
    #    despite the fact that physically point 1 is closer to 0 than 2
    beta = 5
    alpha = 0.01
    actual_label_distance_5_001, mask = sam_label_distance(
        sam_features, spatial_distance, beta, 10, alpha
    )

    # 2) in this case, the most important criterion is physical distance
    #    the distance between points 0 and 1 (1.11) is less than between points 0 and 2 (1.55),
    #    so the value of the label distance between them will be higher (0.0039 versus 0.0004),
    #    despite the fact that the instances of point 0 completely coincide with 2
    beta = 0.01
    alpha = 5
    actual_label_distance_001_5, mask = sam_label_distance(
        sam_features, spatial_distance, beta, 10, alpha
    )

    assert (actual_label_distance_5_001 - expected_label_distance_beta5_alpha001 <= epsilon).all()
    assert (actual_label_distance_001_5 - expected_label_distance_beta001_alpha5 <= epsilon).all()
