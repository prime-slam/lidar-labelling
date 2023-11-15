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

import matplotlib.pyplot as plt
import numpy as np
import random
import supervision as sv

from PIL import Image as im
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from constants import CHECKPOINT_PATH
from constants import DEVICE
from constants import MODEL_TYPE


def generate_random_colors(N):
    colors = [[0, 0, 0]]
    for _ in range(N):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    colors = np.vstack(colors) / 255
    return colors


def get_annotated_image(cam_image):
    sam_result = get_array_instances_by_image_sam(cam_image)
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(sam_result=sam_result)

    black_image = get_black_image(cam_image)

    annotated_image = mask_annotator.annotate(scene=np.array(black_image), detections=detections, opacity=1.0)

    return im.fromarray(annotated_image)


def get_array_instances_by_image_sam(cam_image):
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator.generate(cam_image)


def get_black_image(src_image):
    black_image = im.fromarray(src_image)
    black_pixels = black_image.load()
    for x in range(black_image.width):
        for y in range(black_image.height):
            black_pixels[x, y] = (0, 0, 0)

    return black_image


def visualize_image(image):
    plt.imshow(image)
