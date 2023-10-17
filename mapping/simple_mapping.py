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
import logging
import numpy as np
import open3d as o3d

from logger_message import SUCCESSFUL_IMAGE_PROCESSING
from mapping.abstract_mapping import AbstractMapping
from mapping.coo_matrix_view import CooMatrixView
from utils.image_utils import generate_random_colors
from utils.pcd_utils import remove_hidden_points
from utils.pcd_utils import get_point_map


class SimpleMapping(AbstractMapping):
    def __init__(self, dataset):
        super().__init__(dataset)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('simple_mapping.SimpleMapping')

    def get_combined_labeled_point_clouds(self, cam_name, start_index, end_index):
        pcd_combined = get_point_map(cam_name, self.dataset, start_index, end_index)

        labeled_pcds = []
        labeled_colored_pcds = []
        coo_matrix_list = []
        for current_image_index in range(start_index, end_index):
            pcds_prepared = self.dataset.prepare_points_before_mapping(
                cam_name,
                copy.deepcopy(pcd_combined),
                start_index,
                current_image_index
            )

            pcd_hidden_removal = remove_hidden_points(pcds_prepared)

            labeled_pcd, data = self.segment_instances(pcd_hidden_removal, cam_name, current_image_index)

            labeled_pcds.append(labeled_pcd)
            coo_matrix_list.append(CooMatrixView(data, current_image_index, pcd_hidden_removal))

            current_labeled_pcd_index = len(labeled_pcds) - 1
            labeled_colored_pcds.append(
                self.color_points_by_labels(pcd_hidden_removal, labeled_pcds[current_labeled_pcd_index])
            )

            self.logger.info(SUCCESSFUL_IMAGE_PROCESSING.format(current_image_index))

        return labeled_pcds, labeled_colored_pcds, coo_matrix_list

    def segment_instances(self, pcd, cam_name, image_index):
        masks = self.dataset.get_image_instances(cam_name, image_index)
        image_labels = self.masks_to_image(masks)

        p2pix = self.points_to_pixels(
            cam_name,
            np.asarray(pcd.points),
            (image_labels.shape[1], image_labels.shape[0])
        )

        labels = np.zeros(np.asarray(pcd.points).shape[0])
        data = np.zeros(len(labels))

        for ind, value in p2pix.items():
            labels[ind] = image_labels[value[1], value[0]]
            data[ind] = labels[ind]

        return labels, data

    def masks_to_image(self, masks):
        image_labels = np.zeros(masks[0]['segmentation'].shape)
        for i, mask in enumerate(masks):
            image_labels[mask['segmentation']] = i + 1
        return image_labels

    def points_to_pixels(self, cam_name, points, img_shape):
        img_width, img_height = img_shape

        points_proj = self.dataset.get_camera_intrinsics(cam_name) @ points.T
        points_proj[:2, :] /= points_proj[2, :]
        points_coord = points_proj.T

        inds = np.where(
            (points_coord[:, 0] < img_width) & (points_coord[:, 0] >= 0) &
            (points_coord[:, 1] < img_height) & (points_coord[:, 1] >= 0) &
            (points_coord[:, 2] > 0)
        )[0]

        points_ind_to_pixels = {}
        for ind in inds:
            points_ind_to_pixels[ind] = points_coord[ind][:2].astype(int)

        return points_ind_to_pixels

    def color_points_by_labels(self, pcd, labels):
        random_colors = generate_random_colors(500)

        colors = []
        for i in range(labels.shape[0]):
            colors.append(random_colors[int(labels[i]) + 1])

        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors) / 255)

        return pcd
