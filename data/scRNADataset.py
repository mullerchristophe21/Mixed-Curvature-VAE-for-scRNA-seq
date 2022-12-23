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
import json
import os
import pandas as pd
import numpy as np
from scipy.io import mmread
import torch
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler



# ## parse json config file
# config_file = "./data/adipose/adipose.json"
# configs = json.load(open(config_file, "r"))


# data_file = configs["data_file"]
# batch_files = configs["batch_files"]



class scRNADataset:
    """
    Load a dataset using this class:
        data_file: path to a matrix data file
        batch_files: list of batch effect file(s) 
    1st dimension (number of rows) in data_file and batch file(s) must match
    """

    def __init__(self, data_file: str, batch_files: Optional[list] = None) -> None:
        self.data_file = data_file
        self.batch_files = batch_files
        self.dataset: torch.utils.data.TensorDataset = {}

        if batch_files is not None:
            self.batch_data = self._read_batcheff()
            self.batch_data_dim = self.batch_data.shape[1]
        else:
            self.batch_data = None
            self.batch_data_dim = 0

        self.dataset = self._read_data()
        self.dataset_len = self.__len__()
        self.dataset_dim = self.dataset.shape[1] - self.batch_data_dim
        self._shuffle_split_indx()

    def __len__(self) -> int:
        return self.dataset.size(0)

    def __getitem__(self, index):
        return self.dataset[index, :]

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            return device

    def df_to_tensor(self, df):
        device = self.get_device()
        return torch.from_numpy(df.values).float().to(device)

    def read_mtx(self, filename, dtype='int32'):
        x = mmread(filename).astype(dtype)
        return x

    # read batch effect file(s)
    def _read_batcheff(self):
        if len(self.batch_files) == 1:
            batch_data = pd.read_csv(self.batch_files[0], header=None)
        else:
            list_batch = list()
            for one_file in self.batch_files:
                one_batch = pd.read_csv(one_file, header=None)
                list_batch.append(one_batch)
            batch_data = pd.concat(list_batch, axis=1)
        return batch_data

    # read data and batch effect files
    def _read_data(self):
        data = self.read_mtx(self.data_file).transpose().todense()
        data = pd.DataFrame(data)
        if self.batch_data is not None:
            assert (
                self.batch_data.shape[0] == data.shape[0]
            ), "batch_data.shape: %s, data.shape: %s" % (
                selfbatch_data.shape[0],
                data.shape[0],
            )
            data = pd.concat([data, self.batch_data], axis=1)
        else:
            data = pd.DataFrame(data)
        res = self.df_to_tensor(data)
        return res

    def _shuffle_split_indx(self):
        indices = list(range(self.__len__()))
        split = int(np.floor(0.5 * self.__len__()))
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
        self.train_sampler = SubsetRandomSampler(indices[:split])
        self.test_sampler = SubsetRandomSampler(indices[split:])

    def create_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            # num_workers=8,
            # pin_memory=True,
            sampler=self.train_sampler,
        )
        test_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            # num_workers=8,
            # pin_memory=True,
            sampler=self.test_sampler,
        )
        return train_loader, test_loader


# ee = scRNADataset(configs['data_file'])
# aa = scRNADataset(configs['data_file'], configs['batch_files'])
# bb, cc = aa.create_loaders(batch_size=100)
# ee = aa.create_loaders(batch_size=100)
# type(bb)
