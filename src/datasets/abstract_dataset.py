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

from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self, dataset):
        self._dataset = dataset
        super().__init__()

    @property
    def dataset(self):
        return self._dataset

    @abstractmethod
    def get_point_cloud(self, index):
        pass

    @abstractmethod
    def get_lidar_pose(self, index):
        pass

    @abstractmethod
    def get_camera_names(self):
        pass

    @abstractmethod
    def get_camera_image(self, cam_name, index):
        pass

    @abstractmethod
    def get_image_instances(self, cam_name, index):
        pass

    @abstractmethod
    def get_camera_intrinsics(self, cam_name):
        pass

    @abstractmethod
    def get_camera_extrinsics(self, cam_name):
        pass
