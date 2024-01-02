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

import zope.interface

from services.preprocessing.common.interface import IProcessor
from utils.pcd_utils import remove_statistical_outlier_points


@zope.interface.implementer(IProcessor)
class StatisticalOutlierProcessor:
    def process(self, config, pcd, points2instances):
        """Removing statistical outlier points taking into account neighbors and threshold value from the config"""

        pcd, ind = remove_statistical_outlier_points(
            pcd, config.nb_neighbors, config.std_ratio
        )

        return pcd, points2instances[ind]
