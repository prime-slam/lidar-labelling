import numpy as np
import open3d as o3d


def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def get_point_map(pcd_dataset, start_index, end_index):
    base_cloud_index = start_index

    pcd_combined = o3d.geometry.PointCloud()
    for current_cloud_index in range(start_index + 1, end_index):
        pcd_combined = paired_association(
            pcd_dataset,
            base_cloud_index,
            current_cloud_index,
            pcd_combined
        )

    return pcd_combined


def paired_association(pcd_dataset, target_cloud_index, src_cloud_index, pcd_combined):
    target_cloud = pcd_dataset.get_point_cloud(target_cloud_index)
    src_cloud = pcd_dataset.get_point_cloud(src_cloud_index)

    matrix_src_cloud_to_target = pcd_dataset.calculate_pcd_motion_matrix(src_cloud_index, target_cloud_index)

    src_cloud.transform(matrix_src_cloud_to_target)

    if len(pcd_combined.points) == 0:
        pcd_combined += target_cloud

    pcd_combined += src_cloud

    return pcd_combined


def remove_hidden_points(pcd):
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    )
    camera = [0, 0, 0]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    return pcd.select_by_index(pt_map)
