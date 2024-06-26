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


class IProcessor(zope.interface.Interface):
    def process(distance_matrix, points, trace):
        """Extracting only those points that have certain properties
        based on the distance matrix.

        The result of processing is a distance matrix, a set of points
        and a trace that contains only those points that satisfy the desired property.
        """
