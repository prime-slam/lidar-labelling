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

import csv


def calculate_sum_ones_zeros(values):
    sum, ones, zeros = 0, 0, 0
    for value in values:
        sum += float(value)
        if float(value) == 1.0:
            ones += 1
        elif float(value) == 0.0:
            zeros += 1
    return sum, ones, zeros


def calculate_metrics(file_name):
    values_pres = []
    values_recall = []
    values_fScore = []
    with open(file_name, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            values_pres.append(row["precision"])
            values_recall.append(row["recall"])
            values_fScore.append(row["fScore"])

    sum_pres, pres1, pres0 = calculate_sum_ones_zeros(values_pres)
    sum_recall, recall1, recall0 = calculate_sum_ones_zeros(values_recall)
    sum_fScore, fscore1, fscore0 = calculate_sum_ones_zeros(values_fScore)

    print(
        "precision={}, 1={}, 0={}".format(
            sum_pres / float(len(values_pres)),
            pres1 / float(len(values_pres)),
            pres0 / float(len(values_pres)),
        )
    )
    print(
        "recall={}, 1={}, 0={}".format(
            sum_recall / float(len(values_recall)),
            recall1 / float(len(values_recall)),
            recall0 / float(len(values_recall)),
        )
    )
    print(
        "fscore={}, 1={}, 0={}".format(
            sum_fScore / float(len(values_fScore)),
            fscore1 / float(len(values_fScore)),
            fscore0 / float(len(values_fScore)),
        )
    )


def main():
    calculate_metrics("experiment_2_a5b5_sem_voxel_offset0_T0l025_50.csv")


if __name__ == "__main__":
    main()
