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

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from data.vae_dataset import VaeDataset
from ..components import Component


class FeedForwardVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component], dataset: VaeDataset,
                 scalar_parametrization: bool) -> None:
        super().__init__(h_dim, components, dataset, scalar_parametrization)
        

        #data dimensions
        self.in_dim = dataset.in_dim
        self.h_dim = h_dim
        #empty tensor 
        self.batch_saver = None
        #batch data
        self.batch_data = dataset.get_batch_effect()
        self.batch_data_dim = self.batch_data.shape[1]
        # 1 hidden layer encoder
        self.fc_e0 = nn.Linear(dataset.in_dim , h_dim)
        # 1 hidden layer decoder
        self.fc_d0 = nn.Linear(self.total_z_dim + self.batch_data_dim, h_dim)
        self.fc_logits = nn.Linear(h_dim, dataset.in_dim) 
        # Batch layer for normailzation
        self.batch_norm = nn.BatchNorm1d(self.h_dim)


        


    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)
        #save the batch effect
        self.batch_saver = x[:,-self.batch_data_dim:]
        #forward pass
        x = torch.relu(self.batch_norm(self.fc_e0(x)))
        return x.view(bs, -1)



    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2 
        
        bs = concat_z.size(-2)
        #concat the batch effect to latent space     
        i = 1
        if len(concat_z.shape) > 2:
            self.batch_saver = self.batch_saver.expand(500,self.batch_saver.shape[0],self.batch_saver.shape[1])
            i = 2

        concat_z = torch.cat((concat_z,self.batch_saver),dim=i)

        #forward pass
        x = self.fc_d0(concat_z)
       






        x = torch.relu(x)
        
        x = self.fc_logits(x)

        
        x = x.view(-1, bs, self.in_dim)  # flatten
        return x.squeeze(dim=0)  # in case we're not doing LL estimation
