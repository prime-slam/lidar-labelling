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
import zope.interface

from src.services.preprocessing.common.interface import IProcessor
from src.utils.pcd_utils import get_subpcd
from src.utils.pcd_utils import get_visible_points


@zope.interface.implementer(IProcessor)
class InitInstancesMatrixProcessor:
    def process(self, config, pcd, points2instances=None):
        """Constructing an instance matrix for pcd.

        The pcd is covered with a sequence of images from (config.start_index - config.start_image_index_offset)
        to config.end_index. The images are already segmented at this point.
        """

        points2instances = self.build_points2instances_matrix(
            pcd,
            config.dataset,
            config.cam_name,
            config.start_index - config.start_image_index_offset,
            config.end_index,
        )

        return points2instances

    def build_points2instances_matrix(
        self, map_wc, dataset, cam_name, start_image_index, end_image_index
    ):
        """The map is moved to the camera coordinate system at the moment the current image is taken.
        The function for removing hidden points is called (get_visible_points).

        Cloud points are related to pixels. The instance number of the corresponding pixel is written into
        the points2instances matrix for the point and the current image.

        The action is repeated for all images.
        """

        N = np.asarray(map_wc.points).shape[0]
        points2instances = np.zeros((N, end_image_index - start_image_index), dtype=int)

        for view_id, view in enumerate(range(start_image_index, end_image_index)):
            masks = dataset.get_image_instances(cam_name, view)
            image_labels = self.masks_to_image(masks)

            T = dataset.get_lidar_pose(view)
            T_cam = T @ np.linalg.inv(dataset.get_camera_extrinsics(cam_name))

            map_cc = copy.deepcopy(map_wc).transform(
                np.linalg.inv(T_cam)
            )  # map in camera frame
            indices_visible = get_visible_points(map_cc)
            map_cc_visible = get_subpcd(map_cc, indices_visible)

            points_to_pixels = self.get_points_to_pixels(
                np.asarray(map_cc_visible.points),
                dataset.get_camera_intrinsics(cam_name),
                ((image_labels.shape[1], image_labels.shape[0])),
            )

            for point_id, pixel_id in points_to_pixels.items():
                points2instances[indices_visible[point_id], view_id] = int(
                    image_labels[pixel_id[1], pixel_id[0]]
                )

        return points2instances

    def get_points_to_pixels(self, points, cam_intrinsics, img_shape):
        """The points of the cloud are matched to the pixels of the image"""

        img_width, img_height = img_shape

        points_proj = cam_intrinsics @ points.T
        points_proj[:2, :] /= points_proj[2, :]
        points_coord = points_proj.T

        inds = np.where(
            (points_coord[:, 0] < img_width)
            & (points_coord[:, 0] >= 0)
            & (points_coord[:, 1] < img_height)
            & (points_coord[:, 1] >= 0)
            & (points_coord[:, 2] > 0)
        )[0]

        points_ind_to_pixels = {}
        for ind in inds:
            points_ind_to_pixels[ind] = points_coord[ind][:2].astype(int)

        return points_ind_to_pixels

    def masks_to_image(self, masks):
        """Assigning instance numbers for each mask of a segmented image"""

        image_labels = np.zeros(masks[0]["segmentation"].shape)
        for i, mask in enumerate(masks):
            image_labels[mask["segmentation"]] = i + 1
        return image_labels
