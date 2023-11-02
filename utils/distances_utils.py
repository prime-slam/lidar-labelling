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

from sklearn.metrics.pairwise import pairwise_distances


def calculate_pairwise_distances(enc_coo_non_zero_instances):
    return pairwise_distances(enc_coo_non_zero_instances)


def calculate_distances(inverse_enc_coo_non_zero_instances):
    shape = len(inverse_enc_coo_non_zero_instances)

    # initializing the distance matrix
    distances = np.zeros((shape, shape))

    for first in range(shape):
        if first % 100 == 0:
            print(first)
        # in format: [ 2., 0., 16.]
        instances_first = inverse_enc_coo_non_zero_instances[first]

        distances_from_first_to = np.zeros(shape)
        for second in range(first + 1, shape):
            instances_second = inverse_enc_coo_non_zero_instances[second]

            # check the values of point instances in each view
            score = 0
            count_non_zero = 0
            for view in range(len(instances_second)):
                instance_first = int(instances_first[view])
                instance_second = int(instances_second[view])
                if (instance_first != 0) & (instance_second != 0):
                    count_non_zero += 1
                    if instance_first == instance_second:
                        score += 1
            distances_from_first_to[second] = 0 if score == 0 else (score / count_non_zero)

        # fill in the whole distance row for point first
        distances[first] = distances_from_first_to

    # distance matrix is symmetric, fill under the main diagonal
    for i in range(distances.shape[0]):
        for j in range(i):
            distances[i, j] = distances[j, i]

    return distances


def sam_label_distance(sam_features, spatial_distance, proximity_threshold, beta):
    mask = np.where(spatial_distance <= proximity_threshold)

    # Initialize the distance matrix with zeros
    num_points, num_views = sam_features.shape
    distance_matrix = np.zeros((num_points, num_points))

    # Iterate over rows (points)
    for (point1, point2) in (zip(*mask)):
        view_counter = 0
        for view in range(num_views):
            instance_id1 = sam_features[point1, view]
            instance_id2 = sam_features[point2, view]

            if instance_id1 != 0 and instance_id2 != 0:
                view_counter += 1
                if instance_id1 != instance_id2:
                    distance_matrix[point1, point2] += 1
        if view_counter:
            distance_matrix[point1, point2] /= view_counter
            distance_matrix[point2, point1] = distance_matrix[point1, point2]

    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)
    label_distance = mask * np.exp(-beta * distance_matrix)

    return label_distance, mask


def print_pairwise_distances(distances, start, end):
    for first in range(start, end):
        for second in range(first + 1, end):
            print("distance between {} and {} = {}".format(first, second, distances[first, second]))
