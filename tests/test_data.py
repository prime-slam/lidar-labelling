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

from pcd_dataset.kitti_dataset import KittiDataset
from services.preprocessing.common.config import ConfigDTO

dataset_path = "tests/test_dataset/"
sequence = "00"
image_instances_path = "tests/test_pipeline/vfm-labels/sam/00/"

kitti = KittiDataset(dataset_path, sequence, image_instances_path)

config = ConfigDTO(
    **{
        "dataset": kitti,
        "start_index": 3,
        "end_index": 7,
        "start_image_index_offset": 2,
        "cam_name": "cam2",
        "R": 12,
        "nb_neighbors": 5,
        "std_ratio": 2.0,
        "voxel_size": 1.0,
    }
)
