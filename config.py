import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

IMG_SIZE = 256

BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 100

LEARNING_RATE = 0.002

BETAS = (0.0, 0.99)

Z_DIM = 64

BASE_CHANNELS = 64

W0 = 8
H0 = 8

TARGET_SIZES = [
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]

CHANNELS = [
    512,
    512,
    256,
    128,
    64,
    32,
]

LAMBDA_L1 = 10.0
LAMBDA_PERCEPTUAL = 1.0
LAMBDA_R1 = 10.0

SAVE_EVERY = 5

SAMPLE_EVERY = 2

CHECKPOINT_DIR = "checkpoints"

WANDB_PROJECT = "ViSyn"

NUM_IMAGES = 10397