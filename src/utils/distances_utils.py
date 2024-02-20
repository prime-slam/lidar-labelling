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
import numpy as np


def sam_label_distance(
    sam_features, spatial_distance, proximity_threshold, beta, alpha
):
    """Calculating a matrix of distances between points based on preliminary labeling and physical distance between points.

    Parameters
    ----------
    sam_features : matrix
        instance matrix
    spatial_distance : matrix
        physical distance between cloud points
    proximity_threshold : int
        threshold value of the distance between close points
    beta : int
        parameter to increase the spread of distance values in the final calculation (part with distance between instances)
    alpha : int
        parameter to increase the spread of distance values in the final calculation (part with physical distance)
    """

    mask = np.where(spatial_distance <= proximity_threshold)

    # Initialize the distance matrix with zeros
    num_points, num_views = sam_features.shape
    distance_matrix = np.zeros((num_points, num_points))

    # Iterate over rows (points)
    for point1, point2 in zip(*mask):
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
    label_distance = (
        mask * np.exp(-beta * distance_matrix) * np.exp(-alpha * spatial_distance)
    )

    return label_distance, mask


def remove_isolated_points(dist, points, trace):
    """Removing isolated points that have all 0s in the distance matrix except the diagonal element"""

    mask_isolated = np.all(dist - np.eye(dist.shape[0]) == 0, axis=1)
    isolated_points = np.array([i for i in range(len(points))], dtype=int)[
        mask_isolated
    ]

    trace_copy = copy.deepcopy(trace)
    for index in sorted(isolated_points, reverse=True):
        del trace_copy[index]

    mask_not_isolated = np.any(dist - np.eye(dist.shape[0]) != 0, axis=1)

    return (
        dist[mask_not_isolated][:, mask_not_isolated],
        points[mask_not_isolated],
        trace_copy,
    )

def extract_largest_connected_component(dist, points, trace):
    """Selection the largest connected component of a graph using the dfs algorithm"""

    num_vertices = len(points)
    half_num_vertices = num_vertices // 2
    visited_vertices = np.array([False for i in range(num_vertices)], dtype=bool)
    for i in range(num_vertices):
        visited_vertices = dfs(dist, i)
        if visited_vertices.sum() >= half_num_vertices:
            break

    not_visited_vertices = [ind for ind, vertex in enumerate(visited_vertices) if not vertex]

    trace_copy = copy.deepcopy(trace)
    for index in sorted(not_visited_vertices, reverse=True):
        del trace_copy[index]

    return (
        dist[visited_vertices][:, visited_vertices],
        points[visited_vertices],
        trace_copy,
    )


def dfs(distance_matrix, start_vertex):
    num_vertices = len(distance_matrix)
    visited = np.array([False for i in range(num_vertices)], dtype=bool)
    stack = [start_vertex]

    while stack:
        current_vertex = stack.pop()
        if not visited[current_vertex]:
            visited[current_vertex] = True

            for neighbor in range(num_vertices):
                if distance_matrix[current_vertex][neighbor] > 0 and not visited[neighbor]:
                    stack.append(neighbor)

    return visited
