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

import numpy as np
import zope.interface

from services.preprocessing.common.interface import IProcessor
from utils.pcd_utils import get_subpcd


@zope.interface.implementer(IProcessor)
class SelectionNotZeroProcessor:

    def process(self, config, pcd, points2instances):
        not_zero_indices = self.get_not_zero_mask(points2instances)
        
        pcd_not_zero = get_subpcd(pcd, not_zero_indices)
        points2instances_not_zero = points2instances[not_zero_indices]

        return pcd_not_zero, points2instances_not_zero

    def get_not_zero_mask(self, points2instances):
        return np.any(points2instances != 0, axis=1)
