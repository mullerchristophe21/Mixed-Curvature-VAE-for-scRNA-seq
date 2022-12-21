
from typing import Dict, Tuple, Optional, Any
import numpy as np

import json
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from vae_dataset import VAEDataset


from util.util import read_mtx

class scRNADataset(torch.utils.data.Dataset):

    def __init__(self, directory: str) -> None:
        
        x = read_mtx(directory).transpose().todense()




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
        
