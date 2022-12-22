import torch
import numpy as np
from config import *
from mvae.models import Trainer, FeedForwardVAE
from mvae import utils
from util.stats import EpochStats
from util.plot import plot_trace
import matplotlib.pyplot as plt
from data.scRNADataset import scRNADataset

import json
import os
import datetime

COMPONENTS = utils.parse_components(MODEL,FIXED_CURVATURE)


####################
## parse data json file. one file per each batch size
# {
#  "data_file": "./data/adipose/adipose.mtx",
#  "batch_files": [
#   	"./data/adipose/adipose_batch.tsv",
#   	"./data/adipose/adipose_batch.tsv",
#   	"./data/adipose/adipose_batch.tsv"
#   ]
# }

config_file = "./data/adipose/adipose.json"
configs = json.load(open(config_file, "r"))

# load dataset with batch effect files
# dataset = scRNADataset(data_file = configs["data_file"],
#                       batch_files = configs["batch_files"])
# load dataset without batch effect files

dataset = scRNADataset(data_file = configs["data_file"])
####################

def setup():

    if SEED:
        print("Using pre-set random seed:", SEED)
        utils.set_seed(SEED)
    #setup device
    utils.setup_gpu(DEVICE)
    print("Running on:", DEVICE, flush=True)
    #setup precision
    if DOUBLES:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

def train_model():

    #Track training progress
    print("#####") 
    cur_time = datetime.datetime.utcnow().isoformat()
    print(f"VAE MODEL: Feedforward VAE; Epochs: {EPOCHS}; Time: {cur_time}; Fixed curvature: {FIXED_CURVATURE}"
            f"Dataset: {dataset}")
    print("####")
    chkpt_dir = f"./chkpt/vae - {dataset}-FeedForwardVAE-{cur_time}"
    os.makedirs(chkpt_dir)





    #Load model
    model = FeedForwardVAE(h_dim = H_DIM, components= COMPONENTS,
        dataset=dataset, scalar_parametrization=SCALAR_PARAMETRIZATION)

    #Load trainer
    trainer = Trainer(model,
                    train_statistics= TRAIN_STATISTICS,
                    show_embeddings= SHOW_EMBEDDINGS,
                    export_embeddings= EXPORT_EMBEDDINGS,
                    test_every= TEST_EVERY)

    #Load optimizer
    optimizer = trainer.build_optimizer(learning_rate=LEARNING_RATE,
                                fixed_curvature= FIXED_CURVATURE)

    #Split data into train and test
    # create_loader require the argument batch size: int 
    train_loader, test_loader = dataset.create_loaders(CHANGE_HERE)
    #Beta priors
    betas = utils.linear_betas(BETA_START,BETA_END,
                        end_epoch=BETA_END_EPOCH, epochs= EPOCHS)


    #choose training mode
    if UNIVERSAL:
        # Pre-training:
        trainer.train_epochs(optimizer=optimizer,
                             train_data=train_loader,
                             eval_data=test_loader,
                             epochs= EPOCHS // 2,
                             betas=betas,
                             likelihood_n=0)

        # Choose signs:
        eps = 1e-5
        cn = len(model.components) // 3
        signs = [-1] * cn + [1] * cn + [0] * (len(model.components) - 2 * cn)
        print("Chosen signs:", signs)
        for i, component in enumerate(model.components):
            component._curvature.data += signs[i] * eps
            component._curvature.requires_grad = False

        # ... and continue without learning curvature for an epoch:
        trainer.train_epochs(optimizer=optimizer,
                             train_data=train_loader,
                             eval_data=test_loader,
                             epochs=10,
                             betas=betas,
                             likelihood_n=0)

        # ... then unfix it:
        for component in model.components:
            component._curvature.requires_grad = True
        trainer.train_stopping(optimizer=optimizer,
                               train_data=train_loader,
                               eval_data=test_loader,
                               warmup= LOOKAHEAD + 1,
                               lookahead= LOOKAHEAD,
                               betas=betas,
                               likelihood_n= LIKELIHOOD_N,
                               max_epochs= EPOCHS)
    else:
        trainer.train_stopping(optimizer=optimizer,
                               train_data=train_loader,
                               eval_data=test_loader,
                               warmup=WARMUP,
                               lookahead=LOOKAHEAD,
                               betas=betas,
                               likelihood_n=LIKELIHOOD_N,
                               max_epochs=EPOCHS)
    print(flush=True)
    print("Done.", flush=True)

def eval_model():


    model = FeedForwardVAE(h_dim = H_DIM,
                        components= COMPONENTS,
                        dataset= dataset,
                        scalar_parametrization= SCALAR_PARAMETRIZATION)     

    model.load_state_dict(torch.load(CHKPT, map_location=DEVICE))

    print("Loaded model: FeedForwardVAE at epoch", EPOCHS , "from" , CHKPT)


    _, test_loader = dataset.create_loaders()

    print(f"\tEpoch {EPOCHS}:\t", end="")
    model.eval()

    batch_stats = []
    for batch_idx, (x_mb, y_mb) in enumerate(test_loader):
        x_mb = x_mb.to(model.device)
        reparametrized, concat_z, x_mb_ = model(x_mb)
        stats = model.compute_batch_stats(x_mb, x_mb_, reparametrized, likelihood_n= LIKELIHOOD_N, beta = 1.)
        batch_stats.append(stats.convert_to_float())



    epoch_stats = EpochStats(batch_stats, length = len(test_loader.dataset))
    epoch_dict = epoch_stats.to_print()

    for i, component in enumerate(model.components):
        name = f"{component.summary_name(i)}/curvature"
        epoch_dict[name] = float(component.manifold.curvature)
    print(epoch_dict, flush=True)
    print("Done.", flush=True)
    
    return epoch_dict


def main() -> None:
    #Setup config
    setup()
    #Train the model!
    train_model()
    #Evaluate the model!
    epoch_dict = eval_model()
    """
    do stuff with the epoch dict....
    """


if __name__ == "__main__":
    main()