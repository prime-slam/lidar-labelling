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

from services.preprocessing.init.map import InitMapProcessor

from tests.test_data import config


def test_init_map():
    init_pcd = InitMapProcessor().process(config)

    real_init_map_size = sum(
        len(config.dataset.get_point_cloud(i).points)
        for i in range(config.start_index, config.end_index)
    )

    assert len(init_pcd.points) == real_init_map_size
