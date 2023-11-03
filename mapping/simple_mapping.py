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

from logger_message import SUCCESSFUL_IMAGE_PROCESSING
from mapping.abstract_mapping import AbstractMapping
from mapping.coo_matrix_view import CooMatrixView
from utils.pcd_utils import remove_hidden_points
from utils.pcd_utils import get_point_map


class SimpleMapping(AbstractMapping):
    def __init__(self, dataset):
        super().__init__(dataset)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('simple_mapping.SimpleMapping')

    def get_combined_labeled_point_clouds(self, cam_name, start_index, end_index):
        pcd_combined = get_point_map(cam_name, self.dataset, start_index, end_index)

        pcds = []
        coo_matrix_list = []
        for current_image_index in range(start_index, end_index):
            pcd_prepared, pt_map = self.prepare_pcd_before_mapping(cam_name, pcd_combined, start_index, current_image_index)

            labels_data = self.segment_instances(pcd_combined, pcd_prepared, pt_map, cam_name, current_image_index)

            pcds.append(pcd_prepared)
            coo_matrix_list.append(
                CooMatrixView(labels_data, current_image_index, pcd_prepared)
            )

            self.logger.info(SUCCESSFUL_IMAGE_PROCESSING.format(current_image_index))

        return coo_matrix_list, pcds, pcd_combined

    def prepare_pcd_before_mapping(self, cam_name, pcd_combined, start_index, current_image_index):
        pcds_prepared = self.dataset.prepare_points_before_mapping(
            cam_name,
            copy.deepcopy(pcd_combined),
            start_index,
            current_image_index
        ) # здесь карта приводится к current_image_index-тому снимку, количество точек pcds_prepared совпадает с pcd_combined

        pcd, pt_map = remove_hidden_points(pcds_prepared)

        # pcd, pt_map = pcds_prepared, [] # если закомментировать предыдущую строку и раскомментировать эту, то раскраска карты
                                          # будет то что надо (кроме этих двух строк для успешного результата менять ничего не надо)
        return pcd, pt_map

    def segment_instances(self, pcd_combined, pcd, pt_map, cam_name, image_index):
        # pcd_combined --- карта вообще без преобразований
        # pcd --- подготовленное облако --- приведено к image_index-тому снимку и пропущено через hidden_point_removal
        # pt_map --- мапа--результат hidden_point_removal

        # я думаю, что работает так:
        # pt_map[new_point]=old_point, где new_point --- точка подготовленного pcd,
        # а old_point --- соответствующая точка pcd_combined ПОСЛЕ ПРИВЕДЕНИЯ К image_index-тому СНИМКУ (!!!)
        # мне хочется зацепиться за момент на предыдущей строке, что pt_map это связь НЕ с pcd_combined,
        # а с его приведением к снимку, от этого и проблемы (приведение тут -> AbstractDataset prepare_points_before_mapping)
        # но не похоже на правду, потому что:

        # 1) тупо убираем hidden_point - все работает
        # 2) условно взяли точку pcd_combined с индексом 123. после преобразований меняются ее координаты (?) в сторону снимка,
        # но этой точке 123 в подготовленном облаке все так же соответствует точка 123

        masks = self.dataset.get_image_instances(cam_name, image_index)
        image_labels = self.masks_to_image(masks)

        p2pix = self.points_to_pixels(
            cam_name,
            np.asarray(pcd.points),
            (image_labels.shape[1], image_labels.shape[0])
        )

        labels_data = np.zeros(np.asarray(pcd_combined.points).shape[0])

        print("len(p2pix.items()) = {}".format((len(p2pix))))
        print("len(pt_map) = {}".format(len(pt_map)))
        print("len(pcd_combined.points) = {}".format(len(pcd_combined.points)))
        print("len(pcd.points) = {}".format(len(pcd.points)))

        # тут я убедилась, что список pt_map содержит индексы точек облака после hidden НЕ по порядку от 0 до n-1,
        # а вразброс, то есть список реально содержит СТАРЫЕ индексы, которые видны с данного участка
        # max = -1
        # maxind = -1
        # for i in range(len(pt_map)):
        #     if i > maxind:
        #         maxind = i
        #     if pt_map[i] > max:
        #         max = pt_map[i]
        # print("pt_map: maxvalue = {}. maxind = {}".format(max, maxind))

        # тут я проверила, что в pcd индексы уже по порядку --- максимальный равен количеству точек в облаке --- это НОВЫЕ индексы
        # maxind = -1
        # for i in range(len(pcd.points)):
        #     if i > maxind:
        #         maxind = i
        # print("maxind pcd = {}".format(maxind))

        # тут пыталась сама руками построить мапу <новый индекс в pcd, старый в pcd_combined>,
        # ожидаемо получила тот же плохой результат (ожидаемо, потому что это эквивалентно pt_map[ind] в цикле ниже)
        # map_new_old = {}
        # for point in range(len(pcd_combined.points)):
        #     if point % 1000 == 0:
        #         print(point)
        #     if point in pt_map: # если старый индекс есть в мапе после hidden, то у него есть соответ новый индекс
        #         pair = {pt_map.index(point): point}
        #         map_new_old.update(pair)
        #     else:
        #         pair = {point: point}
        #         map_new_old.update(pair)
        #
        # print("len(map_new_old) = {}".format(len(map_new_old))) # тут длина почему-то ~ на 50-80к меньше, чем pcd_combined.points

        for ind, value in p2pix.items():
            # index = pt_map[ind] # изначально казалось, что решение ровно в одной этой строке, а затем labels_data[index] =, но нет
            labels_data[ind] = image_labels[value[1], value[0]]

        return labels_data

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
