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

import numpy as np

# The name of the .label file for a specific cloud is its number in sequence
# and a few leading zeros so that the number of characters is 6 without extension

GT_FILENAME_LEN = 6
GT_FILE_EXTENSION = ".label"


def build_sem_inst_label_arrays(gt_label_filepath, start_index, end_index):
    filepath = __get_gt_label_filepath_for_pcd(gt_label_filepath, start_index)
    sem_label_array_src, inst_label_array_src = __build_label_arrays_by_file(filepath)

    for i in range(start_index + 1, end_index):
        filepath = __get_gt_label_filepath_for_pcd(gt_label_filepath, i)
        sem_label_array_cur, inst_label_array_cur = __build_label_arrays_by_file(
            filepath
        )

        inst_label_array_src = np.append(inst_label_array_src, inst_label_array_cur)
        sem_label_array_src = np.append(sem_label_array_src, sem_label_array_cur)

    return sem_label_array_src, inst_label_array_src


def combine_sem_inst_labels(sem_label_array, inst_label_array):
    if len(sem_label_array) != len(inst_label_array):
        raise Exception("Array lengths are not equal")

    combined_label_array = np.zeros((len(sem_label_array),), dtype=int)
    for i in range(len(sem_label_array)):
        if inst_label_array[i] != 0:
            combined_label_array[i] = inst_label_array[i]
        else:
            combined_label_array[i] = sem_label_array[i]
    return combined_label_array


def __build_label_arrays_by_file(gt_label_path):
    with open(gt_label_path, "rb") as f:
        label_data = np.fromfile(f, dtype=np.uint32)
        label_data = label_data.reshape((-1))

        sem_label = label_data & 0xFFFF
        inst_label = label_data >> 16

    return sem_label, inst_label


def __get_gt_label_filepath_for_pcd(gt_labels_path, pcd_ind):
    amount_leading_zeros = GT_FILENAME_LEN - len(str(pcd_ind))
    filename = "0" * amount_leading_zeros + str(pcd_ind) + GT_FILE_EXTENSION
    return gt_labels_path + filename
