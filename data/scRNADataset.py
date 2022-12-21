# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, Tuple, Optional, Any
import numpy as np

import json
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from vae_dataset import VAEDataset



class scRNADataset(torch.utils.data.Dataset):

    def __init__(self, dim: int)
        self.dim = int(dim)

        self.data -= np.mean(self.data, axis=0, keepdims=True)
        self.data /= np.std(self.data, axis=0, keepdims=True)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Generates one sample
        """
        data = self.data[idx]
        return torch.Tensor(data)
        
