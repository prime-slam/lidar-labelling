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

import copy

from src.utils.geometry_utils import find_intersection
from src.utils.geometry_utils import find_union


def find_intersection_mask(mask1, mask2):
    bbox = find_intersection(mask1["bbox"], mask2["bbox"])
    if bbox == None:
        return None
    segmentation = mask1["segmentation"] * mask2["segmentation"]
    area = segmentation.sum()

    intersection_mask = copy.deepcopy(mask1)
    intersection_mask["segmentation"] = segmentation
    intersection_mask["bbox"] = bbox
    intersection_mask["area"] = area
    return intersection_mask


def find_union_mask(mask1, mask2):
    segmentation = mask1["segmentation"] + mask2["segmentation"]
    bbox = find_union(mask1["bbox"], mask2["bbox"])
    area = segmentation.sum()

    union_mask = copy.deepcopy(mask1)
    union_mask["segmentation"] = segmentation
    union_mask["bbox"] = bbox
    union_mask["area"] = area
    return union_mask
