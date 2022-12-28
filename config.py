import torch 


DATA_DIR = ""


#CONFIG FOR MVAE

#wheter to use cude or cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#data directory
DATA = "./data"
#batch size
BATCH_SIZE = 100
#learning rate
LEARNING_RATE = 1e-3
#number of epochs
EPOCHS = 500
#number of epochs
WARMUP = 100
#number of epochs
LOOKAHEAD = 50
#model latent space description
MODEL = "h2,s2,e2"
#Universal training scheme
UNIVERSAL  = False
#Hidden layer dimension
H_DIM = 400
#Random seed
SEED = None
#Show embedding every N test runs. Non-postive values mean never. Only effective if test_every > 0
SHOW_EMBEDDINGS = 0
#Export embedding every N test runs. Non-postive values mean never. Only effective if test_every > 0
EXPORT_EMBEDDINGS = 0
#Test every N epochs during training. Non-positive values mean never.
TEST_EVERY = 0
#Show tensorboard statisitc for training
TRAIN_STATISTICS = False
#Use a spheric covariance matrix (single sclar) if true, or ellipic (diagonal covariance) if false
SCALAR_PARAMETRIZATION = False
#Whether to fix curvature to (-1,0,1)
FIXED_CURVATURE = True
#Use float32 or float64, default float32
DOUBLES = False
#Beta-VAE beginning value
BETA_START = 5.0
#Beta-VAE end value
BETA_END = 1.0
#Beta-VAE end epoch (0 to epochs-1)
BETA_END_EPOCH = 1
#How many samples to use for LL estimation. Value 0 disables LL estimation
LIKELIHOOD_N = 500
#DIRECTORY where checkpoints are stored
CHKPT = "~/ETH/DeepLearning/Mixed-scRNA/chkpt"
