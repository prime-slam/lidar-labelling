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
import copy

import numpy as np

from scipy.sparse import coo_array

from utils.image_utils import generate_random_colors
from utils.pcd_utils import remove_hidden_points
from utils.pcd_utils import get_point_map

import open3d as o3d


def construct_coo_matrix_for_multiple_views(coo_matrix_view_list, start_index):
    views_count = len(coo_matrix_view_list)

    row = calculate_row(coo_matrix_view_list, start_index)
    col = calculate_col(coo_matrix_view_list)
    data = np.concatenate([view.data for view in coo_matrix_view_list])

    return coo_array((data, (row, col)), shape=(views_count, len(data)))


def construct_coo_matrix_for_multiple_views_data(coo_matrix_view_list):
    return np.concatenate([view.data for view in coo_matrix_view_list])


def calculate_row(coo_matrix_view_list, start_index):
    row_list = []
    for view in coo_matrix_view_list:
        if view.image_index == start_index:
            row_list.append(np.zeros((len(view.data),), dtype=int))
        else:
            row_list.append(
                np.array([view.image_index - start_index for i in range(len(view.data))], dtype=int)
            )

    return np.concatenate([row for row in row_list])


def calculate_col(coo_matrix_view_list):
    col_list = []
    for view in coo_matrix_view_list:
        col_list.append(np.array([i for i in range(len(view.data))], dtype=int))

    return np.concatenate([col for col in col_list])


def visualize_labeled_clouds(coo_matrix_view_list, view_num, start_index, end_index, simpleMapping):
    coo_matrix_view_list = coo_matrix_view_list.tocsr()
    current_image_index = view_num
    cam_name = 'cam2'

    pcd_combined = get_point_map(cam_name, simpleMapping.dataset, start_index, end_index)
    pcds_prepared = simpleMapping.dataset.prepare_points_before_mapping(
        cam_name,
        copy.deepcopy(pcd_combined),
        start_index,
        current_image_index
    )
    pcd_hidden_removal = remove_hidden_points(pcds_prepared)

    coo_matrix = coo_matrix_view_list[[view_num - start_index],:]

    random_colors = generate_random_colors(500)

    colors = []
    for i in range(coo_matrix.nnz):
        colors.append(random_colors[int((coo_matrix_view_list[[view_num - start_index], [i]])[0]) + 1])

    pcd_hidden_removal.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)


    return pcd_hidden_removal
