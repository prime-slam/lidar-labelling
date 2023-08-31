from abc import ABC, abstractmethod


class AbstractMapping(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def points_to_pixels(self, pcd_dataset, points, img_shape, image):
        pass

    @abstractmethod
    def get_combined_labeled_point_clouds(self, pcd_dataset, start_index, end_index):
        pass
