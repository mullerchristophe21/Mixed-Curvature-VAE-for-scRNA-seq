import torch
import numpy as np
from config import *
from mvae.models import Trainer, FeedForwardVAE
from mvae import utils
from util.util import read_mtx
from util.plot import plot_trace
import matplotlib.pyplot as plt


import os
import datetime

COMPONENTS = utils.parse_components(MODEL,FIXED_CURVATURE)
dataset=None

def main() -> None:
    #Check seed 
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


    """
    TODO: Create dataset and load it
    
    dataset = create_dataset(dataset, BATCH_SIZE, DATA)
    
    """
   

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
    train_loader, test_loader = dataset.create_loaders()
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


"""
    #embedding all the data
    TODO
    save_path = None
    TODO
    
    
    batch = None
    
    
    #Choose a datapoint to encode 
    x = None
    z_mean = model.encode(x)
    
    
    #Save encoded value
    np.savetxt(save_path +
           'cd14_mono_eryth_latent_250epoch.tsv',
           z_mean)
    """

if __name__ == "__main__":
    main()