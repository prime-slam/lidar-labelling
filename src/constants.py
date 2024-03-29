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

import torch

from pathlib import Path

CHECKPOINT_PATH = Path.cwd().joinpath("weights", "sam_vit_h_4b8939.pth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "vit_h"
