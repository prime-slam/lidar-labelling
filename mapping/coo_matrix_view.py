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

class CooMatrixView:
    def __init__(self, data, image_index, pcd):
        self._data = data
        self._image_index = image_index
        self._pcd = pcd

    @property
    def data(self):
        return self._data

    @property
    def image_index(self):
        return self._image_index

    @property
    def pcd(self):
        return self._pcd
