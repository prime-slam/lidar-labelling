import pykitti
import cv2
import numpy as np
import open3d as o3d

from pcd_dataset.abstract_pcd_dataset import AbstractDataset


class KittiDataset(AbstractDataset):

    def __init__(self, dataset_path, sequence):
        super().__init__(pykitti.odometry(dataset_path, sequence))
        self._poses = self.dataset.poses
        self._T_lidar2cam = self.dataset.calib.T_cam2_velo
        self._cam_intrinsics = self.dataset.calib.K_cam2

    @property
    def poses(self):
        return self._poses

    @property
    def T_lidar2cam(self):
        return self._T_lidar2cam

    @property
    def cam_intrinsics(self):
        return self._cam_intrinsics

    def get_point_cloud(self, index):
        points = self.dataset.get_velo(index)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def get_image(self, index):
        image = self.dataset.get_cam2(index)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def calculate_pcd_motion_matrix(self, src_index, target_index):
        target_cloud_poses = self.poses[target_index]
        src_cloud_poses = self.poses[src_index]

        src_cam_to_target_poses = np.linalg.inv(target_cloud_poses) @ src_cloud_poses
        matrix_src_cloud_to_target = np.linalg.inv(self.T_lidar2cam) @ src_cam_to_target_poses @ self.T_lidar2cam

        return matrix_src_cloud_to_target

    def transform_to_pcd0(self, pcd=None, start_pcd_index=0):
        matrix = np.linalg.inv(self.T_lidar2cam) @ self.poses[start_pcd_index] @ self.T_lidar2cam

        return pcd.transform(matrix)

    def transform_pcd0_to_cami_coordinate_system(self, pcd=None, i=0):
        matrix = np.linalg.inv(self.poses[i]) @ self.T_lidar2cam

        return pcd.transform(matrix)

    def prepare_points_before_mapping(self, pcd, start_pcd_index, image_index):
        pcd_L0 = self.transform_to_pcd0(pcd=pcd, start_pcd_index=start_pcd_index)
        pcd_Ki = self.transform_pcd0_to_cami_coordinate_system(pcd=pcd_L0, i=image_index)

        return pcd_Ki
