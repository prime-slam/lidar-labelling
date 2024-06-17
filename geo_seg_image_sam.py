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

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.constants import CHECKPOINT_PATH
from src.constants import DEVICE
from src.constants import MODEL_TYPE


def get_array_instances_by_image_sam(cam_image):
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator.generate(cam_image)


def main():
    n = 43

    for i in range(n):
        file_name_read = ''
        file_name_write = ''

        if (i <= 9):
            file_name_read = 'images_cams/images/0000000{}.jpg'.format(i)
            file_name_write = 'geo-seg/vfm-labels/sam/0000000{}.npz'.format(i)
        else:
            file_name_read = 'images_cams/images/000000{}.jpg'.format(i)
            file_name_write = 'geo-seg/vfm-labels/sam/000000{}.npz'.format(i)

        instances_array = get_array_instances_by_image_sam(cv2.imread(file_name_read))

        np.savez_compressed(file_name_write, masks=instances_array)
        print("Finish i={}".format(i))


if __name__ == "__main__":
    main()
