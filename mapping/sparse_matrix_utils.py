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
import open3d as o3d

from scipy.sparse import coo_array

from utils.image_utils import generate_random_colors


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


def visualize_labeled_clouds(coo_matrix_views, pcds, start_index, view_num):
    coo_matrix = coo_matrix_views.tocsr()[[view_num - start_index], :]

    random_colors = generate_random_colors(500)

    colors = []
    for i in range(coo_matrix.nnz):
        colors.append(random_colors[int(coo_matrix[0, i])])

    pcd = pcds[view_num - start_index]
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)

    return pcd
