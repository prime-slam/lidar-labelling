from abc import ABC, abstractmethod


class AbstractPointCloudDataset(ABC):

    def __init__(self, dataset):
        self._dataset = dataset
        super().__init__()

    def get_dataset(self):
        return self._dataset

    dataset = property(get_dataset)

    @abstractmethod
    def get_point_cloud(self, index):
        pass

    @abstractmethod
    def get_image(self, index):
        pass

    @abstractmethod
    def calculate_pcd_motion_matrix(self, src_index, target_index):
        pass

    @abstractmethod
    def prepare_points_before_mapping(self, pcd, start_pcd_index, image_index):
        pass
