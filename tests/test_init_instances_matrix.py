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

import open3d as o3d
import pytest

from services.preprocessing.init.instances_matrix import InitInstancesMatrixProcessor

from tests.test_data import config
from tests.test_data import real_image_count
from tests.utils import generate_init_pcd

@pytest.mark.parametrize("init_pcd", [generate_init_pcd(config)])
def test_init_map(init_pcd : o3d.geometry.PointCloud):
    points2instances = InitInstancesMatrixProcessor().process(config, init_pcd)

    assert points2instances.shape == (len(init_pcd.points), real_image_count)
