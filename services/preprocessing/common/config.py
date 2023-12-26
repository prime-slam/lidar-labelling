# Copyright (c) 2023, Sofya Vivdich and Anastasiia Kornilova
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

import attr

from pcd_dataset.abstract_pcd_dataset import AbstractDataset


def is_positive(instance, attribute, value):
   if value <= 0:
      raise ValueError("{} has to be positive!".format(attribute.name))


def is_valid_cam_name(instance, attribute, value):
   if value != 'cam0' and value != 'cam1' and value != 'cam2' and value != 'cam3':
      raise ValueError("{} has to be in [cam0, cam1, cam2, cam3]!".format(attribute.name))


def end_more_than_start(instance, attribute, value):
   if value <= instance.start_index:
      raise ValueError("'end_index' has to be more than 'start_index'!")


def image_offset_less_than_start_index(instance, attribute, value):
   if value > instance.start_index:
      raise ValueError("'start_image_index_offset' has to be less than 'start_index'!")


@attr.s
class ConfigDTO:
   dataset: AbstractDataset = attr.ib()

   start_index: int = attr.ib(default=5, validator=[attr.validators.instance_of(int), is_positive])
   end_index: int = attr.ib(default=10, validator=[attr.validators.instance_of(int), end_more_than_start])
   start_image_index_offset: int = attr.ib(
      default=3, validator=[attr.validators.instance_of(int), is_positive, image_offset_less_than_start_index]
   )
   
   cam_name: str = attr.ib(default='cam2', validator=[attr.validators.instance_of(str), is_valid_cam_name])

   R: int = attr.ib(default=12, validator=[attr.validators.instance_of(int), is_positive])

   nb_neighbors: int = attr.ib(default=25, validator=[attr.validators.instance_of(int), is_positive])
   std_ratio: float = attr.ib(default=5.0, validator=[attr.validators.instance_of(float), is_positive])

   voxel_size: float = attr.ib(default=0.3, validator=[attr.validators.instance_of(float), is_positive])
