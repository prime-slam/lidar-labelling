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


def find_intersection(bbox1, bbox2):
    x1_bbox1, y1_bbox1, w_bbox1, h_bbox1 = bbox1
    x1_bbox2, y1_bbox2, w_bbox2, h_bbox2 = bbox2

    x2_bbox1 = x1_bbox1 + w_bbox1
    y2_bbox1 = y1_bbox1 + h_bbox1
    x2_bbox2 = x1_bbox2 + w_bbox2
    y2_bbox2 = y1_bbox2 + h_bbox2

    if (
        x1_bbox1 > x2_bbox2
        or x2_bbox1 < x1_bbox2
        or y1_bbox1 > y2_bbox2
        or y2_bbox1 < y1_bbox2
    ):
        return None

    x_left = max(x1_bbox1, x1_bbox2)
    x_right = min(x2_bbox1, x2_bbox2)
    y_top = max(y1_bbox1, y1_bbox2)
    y_bottom = min(y2_bbox1, y2_bbox2)

    return [x_left, y_top, x_right - x_left, y_bottom - y_top]


def find_union(bbox1, bbox2):
    x1_bbox1, y1_bbox1, w_bbox1, h_bbox1 = bbox1
    x1_bbox2, y1_bbox2, w_bbox2, h_bbox2 = bbox2

    x2_bbox1 = x1_bbox1 + w_bbox1
    y2_bbox1 = y1_bbox1 + h_bbox1
    x2_bbox2 = x1_bbox2 + w_bbox2
    y2_bbox2 = y1_bbox2 + h_bbox2

    x_left = min(x1_bbox1, x1_bbox2)
    y_top = min(y1_bbox1, y1_bbox2)
    x_right = max(x2_bbox1, x2_bbox2)
    y_bottom = max(y2_bbox1, y2_bbox2)

    return [x_left, y_top, x_right - x_left, y_bottom - y_top]


def calculate_area(bbox):
    w = bbox[2]
    h = bbox[3]
    return w * h
