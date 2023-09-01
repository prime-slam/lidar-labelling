from abc import ABC, abstractmethod


class AbstractMapping(ABC):
    def __init__(self, pcd_dataset):
        self._pcd_dataset = pcd_dataset
        super().__init__()

    def get_pcd_dataset(self):
        return self._pcd_dataset

    pcd_dataset = property(get_pcd_dataset)

    @abstractmethod
    def points_to_pixels(self, points, img_shape, image):
        pass

    @abstractmethod
    def get_combined_labeled_point_clouds(self, start_index, end_index):
        pass
