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
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.vae_dataset import VaeDataset
from torch.distributions import Normal



class scRNADataset(VaeDataset):
    """
    Load a dataset using this class:
        data_file: path to a matrix data file
        label_file: path to a label data file
        batch_files: list of batch effect file(s) 
    1st dimension (number of rows) in data_file and batch file(s) must match
    1st dimension (number of rows) in data_file and label file must match
    batch data available as self.batch_data
    batch data dimensions as self.batch_data_dim
    """

    def __init__(self, batch_size: int, data_folder: str, data_file: str,
                 label_file: str, batch_files: Optional[list] = None,
                doubles = False) -> None:
        # input file paths
        self.data_file = data_file
        self.batch_files = batch_files
        self.label_file = label_file
        self.doubles = doubles
        # read batch effect, labels, data
        if batch_files is not None:
            self.batch_data_pd = self._read_batcheff()
            self.batch_data = self.df_to_tensor(df = self.batch_data_pd)
        else:   
            self.batch_data_pd = self.create_dummy_batches()
            self.batch_data = self.df_to_tensor(df = self.batch_data_pd)
                 
        self.batch_data_dim = self.batch_data.shape[1]

        self.data = self._read_data()
        self.labels, self.label_names = self._read_label()
        
        self.check_dim1()
        self.dataset = torch.utils.data.TensorDataset(self.data,self.labels)
        self.dataset_len = self.__len__()
        self._in_dim = self.dataset.tensors[0].shape[1] #FIXME should in_dim has all the dimensions including batch effect dimensions?
        self._shuffle_split_indx()

        super().__init__(batch_size, img_dims=None, in_dim=self._in_dim) #FIXME: correct dimensions? -Colin
        self.data_folder = data_folder

    def __len__(self) -> int:
        return self.dataset.tensors[0].size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one sample
        """
        data, labels = self.dataset[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            return device

    def df_to_tensor(self, df):
        device = self.get_device()
        if self.doubles:
            res = torch.from_numpy(df.values).double().to(device)
        else:
            res = torch.from_numpy(df.values).float().to(device)
        return res

    def read_mtx(self, filename, dtype='int32'):
        x = mmread(filename).astype(dtype)
        return x

    def get_batch_effect(self):
        return self.batch_data
    
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

    # read label
    def _read_label(self):
        label_names = pd.read_csv(self.label_file, header=None)
        label_as_fct = pd.DataFrame(pd.factorize(label_names[0])[0]+1)
        label_data = self.df_to_tensor(label_as_fct)
        return label_data, label_names

    # read data and batch effect files
    def _read_data(self):
        data = self.read_mtx(self.data_file).transpose().todense()
        data = np.true_divide(data, data.max())
        data = pd.DataFrame(data)
        data = pd.concat([data, self.batch_data_pd], axis=1)
        res = self.df_to_tensor(data)
        return res

    def create_dummy_batches(self):
        dummy = np.full(shape=(self.data.shape[0],1),
                        fill_value=0,
                        dtype='int32')
        dummy = pd.DataFrame(dummy)
        #dummy = self.df_to_tensor(dummy)
        return dummy
    
    def check_dim1(self):
        assert (self.batch_data.shape[0] == self.data.shape[0]
            ), "batch_data.shape: %s, data.shape: %s" % (
                self.batch_data.shape[0], self.data.shape[0],
            )
        assert (self.labels.shape[0] == self.data.shape[0]
            ), "labels.shape: %s, data.shape: %s" % (
                self.labels.shape[0], self.data.shape[0],
            )


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
        
    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return -Normal(x_mb_, 0.5*torch.ones_like(x_mb_)).log_prob(x_mb) #FIXME: negative binomial loss -Colin


#########################################################
## a new implementation. provide batch effect as additional tensor
########################################################


# class scRNADataset2(VaeDataset):
#     """
#     Load a dataset using this class:
#         data_file: path to a matrix data file
#         label_file: path to a label data file
#         batch_files: list of batch effect file(s) 
#     1st dimension (number of rows) in data_file and batch file(s) must match
#     1st dimension (number of rows) in data_file and label file must match
    
#     """

#     def __init__(self, batch_size: int, data_folder: str, data_file: str, label_file: str, batch_files: Optional[list] = None) -> None:
#         # input file paths
#         self.data_file = data_file
#         self.batch_files = batch_files
#         self.label_file = label_file
#         # read data, labels and batch effects
#         self.data = self._read_data()
#         self.labels, self.label_names = self._read_label()
#         if batch_files is not None:
#             self.batches = self._read_batcheff()
#         else:
#             self.batches = self.create_dummy_batches()
#         self.batches_dim = self.batches.shape[1]
#         self.check_dim1()

#         self.dataset = torch.utils.data.TensorDataset(self.data, self.labels, self.batches)
#         self.dataset_len = self.__len__()
#         self._in_dim = self.dataset.tensors[0].shape[1]
#         self._shuffle_split_indx()

#         super().__init__(batch_size, img_dims=None, in_dim=self._in_dim) #FIXME: correct dimensions? -Colin
#         self.data_folder = data_folder
    
#     def __len__(self) -> int:
#         return self.dataset.tensors[0].size(0)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Get one observation
#         """
#         data, labels, batches = self.dataset[idx], self.labels[idx], self.batches[idx]
#         return torch.Tensor(data), torch.Tensor(labels), torch.Tensor(batches)

#     def get_device(self):
#         if torch.cuda.is_available():
#             device = torch.device("cuda:0")
#         else:
#             device = torch.device("cpu")
#             return device

#     def df_to_tensor(self, df):
#         device = self.get_device()
#         return torch.from_numpy(df.values).float().to(device)

#     def read_mtx(self, filename, dtype='int32'):
#         x = mmread(filename).astype(dtype)
#         return x

#     # read batch effect file(s)
#     def _read_batcheff(self):
#         if len(self.batch_files) == 1:
#             batch_data = pd.read_csv(self.batch_files[0], header=None)
#         else:
#             list_batch = list()
#             for one_file in self.batch_files:
#                 one_batch = pd.read_csv(one_file, header=None)
#                 list_batch.append(one_batch)
#             batch_data = pd.concat(list_batch, axis=1)
#             batch_data = self.df_to_tensor(batch_data)
#         return batch_data
    
#     def create_dummy_batches(self):
#         dummy = np.full(shape=(self.data.shape[0],1),
#                         fill_value=0,
#                         dtype='int32')
#         dummy = pd.DataFrame(dummy)
#         dummy = self.df_to_tensor(dummy)
#         return dummy
    
#     # read label
#     def _read_label(self):
#         label_names = pd.read_csv(self.label_file, header=None)
#         label_as_fct = pd.DataFrame(pd.factorize(label_names[0])[0]+1)
#         label_data = self.df_to_tensor(label_as_fct)
#         return label_data, label_names
    

#     # read data and batch effect files
#     def _read_data(self):
#         data = self.read_mtx(self.data_file).transpose().todense()
#         data = np.true_divide(data, data.max())
#         data = pd.DataFrame(data)
#         res = self.df_to_tensor(data)
#         return res
    
#     def check_dim1(self):
#         assert (self.batches.shape[0] == self.data.shape[0]
#             ), "batches.shape: %s, data.shape: %s" % (
#                 self.batches.shape[0], self.data.shape[0],
#             )
#         assert (self.labels.shape[0] == self.data.shape[0]
#             ), "labels.shape: %s, data.shape: %s" % (
#                 self.labels.shape[0], self.data.shape[0],
#             )


#     def _shuffle_split_indx(self):
#         indices = list(range(self.__len__()))
#         split = int(np.floor(0.5 * self.__len__()))
#         # np.random.seed(random_seed)
#         np.random.shuffle(indices)
#         self.train_sampler = SubsetRandomSampler(indices[:split])
#         self.test_sampler = SubsetRandomSampler(indices[split:])

#     def create_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
#         train_loader = DataLoader(
#             dataset=self.dataset,
#             batch_size=batch_size,
#             # num_workers=8,
#             # pin_memory=True,
#             sampler=self.train_sampler,
#         )
#         test_loader = DataLoader(
#             dataset=self.dataset,
#             batch_size=batch_size,
#             # num_workers=8,
#             # pin_memory=True,
#             sampler=self.test_sampler,
#         )
#         return train_loader, test_loader
        
#     def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
#         return -Normal(x_mb_, 10*torch.ones_like(x_mb_)).log_prob(x_mb) #FIXME: negative binomial loss -Colin
