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
import csv
import numpy as np
import pickle

from evops.metrics import precision
from evops.metrics import recall
from evops.metrics import fScore


def find_num_in_inst_label_array(src_points, inst_label_array_for_clustering):
    for point in src_points:
        if inst_label_array_for_clustering[point] > 0:
            return inst_label_array_for_clustering[point]
    return -1


def build_pred_inst_array(
    inst_label_array_for_clustering, clusters, trace, instance_threshold
):
    pred_inst_array = np.zeros(len(inst_label_array_for_clustering), dtype=int)
    free_id = 1
    for cluster in clusters:
        voxel_not_in_gt_cluster_count = 0
        for voxel in cluster:
            src_points = trace[voxel]
            id = find_num_in_inst_label_array(
                src_points, inst_label_array_for_clustering
            )
            if id == -1:
                voxel_not_in_gt_cluster_count += 1

        cluster_in_gt_instance = (
            (len(cluster) - voxel_not_in_gt_cluster_count) / len(cluster)
        ) * 100
        if cluster_in_gt_instance >= instance_threshold:
            for voxel in cluster:
                src_points = trace[voxel]
                for src_point in src_points:
                    pred_inst_array[src_point] = free_id
            free_id += 1
    return pred_inst_array


def convert_clusters_to_list_of_point_arrays(clusters_arr):
    # labels 0 and -1 mean noise, they are equal for us
    for ind, label in enumerate(clusters_arr):
        if label == -1:
            clusters_arr[ind] = 0

    # initializing a list by the number of unique clusters
    clusters_list = []
    for label in set(clusters_arr):
        clusters_list.append([])

    for ind, label in enumerate(clusters_arr):
        clusters_list[label].append(ind)

    # converting internal lists to arrays
    for i in range(len(clusters_list)):
        clusters_list[i] = np.asarray(clusters_list[i])
    return clusters_list


def main():

    from_num = 0
    to_num = 4540

    instance_thresholds = [5, 20, 30, 50]

    for instance_threshold in instance_thresholds:
        print("Start to process instance_threshold={}".format(instance_threshold))

        current_from_num = from_num

        skipped = 0
        while current_from_num < to_num:
            start_index = current_from_num
            end_index = start_index + 4

            file_name = "experiment_bin_0704_4_sem_voxel_offset0_T0l03_hdbscan/start{}_end{}.pickle".format(
                start_index, end_index
            )

            with open(file_name, "rb") as file:
                data = pickle.load(file)

            trace = data[1]["hdbscan_voxel_trace_original"]
            clusters_array = data[2]["hdbscan_clusters"]
            inst_label_array_for_clustering = data[3]["inst_label_array_for_clustering"]

            if (
                inst_label_array_for_clustering.sum() == 0
            ):  # в облаке нет инстансов => пропускаем
                skipped += 1
                print(
                    "start_index={}, end_index={} skip".format(
                        start_index, end_index
                    )
                )
                current_from_num = end_index
                continue

            clusters_list_without_noise = convert_clusters_to_list_of_point_arrays(clusters_array)[1:]

            pred_inst_array = build_pred_inst_array(
                copy.deepcopy(inst_label_array_for_clustering),
                copy.deepcopy(clusters_list_without_noise),
                copy.deepcopy(trace),
                instance_threshold,
            )

            pred_labels = pred_inst_array
            gt_labels = inst_label_array_for_clustering
            tp_condition = "iou"
            precision_res = precision(pred_labels, gt_labels, tp_condition)
            recall_res = recall(pred_labels, gt_labels, tp_condition)
            fScore_res = fScore(pred_labels, gt_labels, tp_condition)

            gt_labels_unique = set(gt_labels)
            gt_labels_unique.discard(0)

            pred_labels_unique = set(pred_labels)
            pred_labels_unique.discard(0)

            with open(
                "experiment_1004_4_without0_sem_voxel_offset0_T0l03_hdbscan_{}.csv".format(
                    instance_threshold
                ),
                "a",
                newline="",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        str(start_index),
                        str(end_index),
                        str(precision_res),
                        str(recall_res),
                        str(fScore_res),
                        len(gt_labels_unique),
                        len(pred_labels_unique),
                        len(clusters_list_without_noise),
                    ]
                )

            print("start_index={}, end_index={} done".format(start_index, end_index))

            current_from_num = end_index

        print(skipped)
        print("Finish to process instance_threshold={}".format(instance_threshold))


if __name__ == "__main__":
    main()
