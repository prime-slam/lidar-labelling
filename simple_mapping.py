import copy
import cv2
import numpy as np
import open3d as o3d

from abstract_mapping import AbstractMapping

from image_utils import get_annotated_image

# from pcd_utils import hidden_removal
from pcd_utils import get_point_map

from constants import VOXEL_SIZE


class SimpleMapping(AbstractMapping):

    def __init__(self):
        super().__init__()

    def points_to_pixels(self, pcd_dataset, points, img_shape, image):
        img_width, img_height = img_shape

        points_proj = pcd_dataset.cam_intrinsics @ points.T
        points_proj[:2, :] /= points_proj[2, :]
        points_coord = points_proj.T

        pixels = image.load()

        inds = np.where(
            (points_coord[:, 0] < img_width) & (points_coord[:, 0] >= 0) &
            (points_coord[:, 1] < img_height) & (points_coord[:, 1] >= 0) &
            (points_coord[:, 2] > 0)
        )[0]

        points_ind_to_pixels = {}
        points_colors = {}
        for ind in inds:
            points_ind_to_pixels[ind] = points_coord[ind][:2].astype(int)
            x = points_ind_to_pixels[ind][0]
            y = points_ind_to_pixels[ind][1]
            points_colors[ind] = pixels[x, y]

        return points_ind_to_pixels, points_colors

    def get_combined_labeled_point_clouds(self, pcd_dataset, start_index, end_index):
        map = get_point_map(pcd_dataset, start_index, end_index)
        pcd_combined_down = map.voxel_down_sample(VOXEL_SIZE)

        # pcd_hidden_removal = hidden_removal(pcd_combined_down) - todo

        pcd_hidden_removal = pcd_combined_down

        labeled_pcds = []
        for current_image_index in range(start_index, end_index):
            image_from_dataset = pcd_dataset.get_image(current_image_index)
            annotated_image = get_annotated_image(image_from_dataset)

            pcds_copy = copy.deepcopy(pcd_hidden_removal)
            pcds_prepared = pcd_dataset.prepare_points_before_mapping(pcds_copy, start_index, current_image_index)

            cam_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

            img_shape = cam_image.shape[1], cam_image.shape[0]

            p2pix, colors = self.points_to_pixels(pcd_dataset, np.asarray(pcds_prepared.points), img_shape, annotated_image)

            pcd_cut = o3d.geometry.PointCloud()
            pcd_cut.points = o3d.utility.Vector3dVector(np.asarray(pcds_prepared.points)[list(p2pix.keys())])
            pcd_cut.colors = o3d.utility.Vector3dVector(np.array(list(colors.values())) / 255)

            o3d.io.write_point_cloud(
                f"annotated_pcds/clouds_{start_index}_{end_index}_im{current_image_index}.pcd",
                pcd_cut
            )

            labeled_pcds.append(pcd_cut)

            print(f"image {current_image_index} processed")

        return labeled_pcds
