from argparse import Namespace
from typing import Optional

from yacs.config import CfgNode as CN

from Prithvi_global_v1.utils.config import check_args, update_common_args


_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to train dataset
_C.DATA.TRAIN_DIR = ''
# Path to valid dataset
_C.DATA.VAL_DIR = ''
# Input image size
_C.DATA.INPUT_SIZE = [4, 224, 224]
# Spectral bands to use
_C.DATA.BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
# Path to json meta file
_C.DATA.META_FILE_PATH = ''
# Data mean per band
_C.DATA.MEAN = [0.0] * 6
# Data std per band
_C.DATA.STD = [1.0] * 6
# Method for scaling data for training ('standard', 'norm_clip', 'log1p' or 'none')
_C.DATA.SCALING = 'standard'

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 1
# Prefetch factor for DataLoader
_C.DATA.PREFETCH_FACTOR = 2

# Mask ratio
_C.DATA.MASK_RATIO = 0.75

# Whether to use data augmentation (RandomCropResize + RandomFlip)
_C.DATA.DATA_AUGMENTATION = False

# Chunk size to be used when loading zarr files (it should be a multiple of the zarr chunk size)
_C.DATA.CHUNK_SIZE = 64

# Don't use threads for Blosc (zarr decompression)
_C.DATA.NO_BLOSC_THREADS = True


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
_C.MODEL.PATCH_SIZE = [1, 14, 14]
_C.MODEL.EMBED_DIM = 1280
_C.MODEL.DEPTH = 32
_C.MODEL.NUM_HEADS = 16
_C.MODEL.DECODER_EMBED_DIM = 512
_C.MODEL.DECODER_DEPTH = 8
_C.MODEL.DECODER_NUM_HEADS = 16
_C.MODEL.MLP_RATIO = 4.
_C.MODEL.NORM_PIX_LOSS = False
# Which type of coords encoding to include ('time' and/or 'location')
_C.MODEL.COORDS_ENCODING = []
# Probability of temporal encoding being dropped
_C.MODEL.COORDS_DROP_RATE = 0.0
# Whether scaling of coords embeddings should be learned
_C.MODEL.COORDS_SCALE_LEARN = False
# Probability of dropping input channels
_C.MODEL.DROP_CHANNELS_RATE = 0.0


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Batch size for a single GPU
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.EPOCHS = 400
_C.TRAIN.WARMUP_EPOCHS = 40
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.LR = 2.4e-3   # lr = base_lr x batch_size / 256;  batch size=4096, base_lr=1.5e-4 -> lr=2.4e-3
_C.TRAIN.WARMUP_LR = 0.
_C.TRAIN.MIN_LR = 0.
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 0.0    # no clip_grad; previous value = 1.0
# Gradient accumulation steps
_C.TRAIN.ACCUMULATION_STEPS = 1

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.95)

# Whether to enable pytorch amp
_C.TRAIN.ENABLE_AMP = False
# Whether to use torch.float16 or torch.bfloat16 for pytorch amp
_C.TRAIN.AMP_DTYPE = 'torch.float16'
# Frequency to save checkpoint (steps)
_C.TRAIN.SAVE_FREQ = 10
# Frequency to log/print info (steps)
_C.TRAIN.PRINT_FREQ = 10
# World size
_C.TRAIN.WORLD_SIZE = 0
# Path to the root directory where to save experiment
_C.TRAIN.BASE_OUTPUT_DIR = ''
# Experiment ID - defined in runtime
_C.TRAIN.EXP_ID = ''
# Path to the directory generated to save experiment (based on EXP_ID)
_C.TRAIN.OUTPUT_DIR = ''
# Checkpoint to resume training from
_C.TRAIN.RESUME = ''
# Checkpoint to load weights from (optimizer/scheduler not loaded)
_C.TRAIN.LOAD_WEIGHTS = ''
# Whether to use only CPU (no GPUs)
_C.TRAIN.CPU_ONLY = False
# Fixed random seed
_C.TRAIN.SEED = 2022


def update_config(config: CN, args: Namespace):
    """ Update config with values from command line args (only if values != None)"""

    config.defrost()
    update_common_args(args, config)

    if check_args('model_name', args):
        config.MODEL.NAME = args.model_name
    if check_args('patch_size', args):
        config.MODEL.PATCH_SIZE = args.patch_size

    # Override configs based on model name
    if config.MODEL.NAME == 'vit_s':
        config.MODEL.EMBED_DIM = 384
        config.MODEL.DEPTH = 12
        config.MODEL.NUM_HEADS = 6
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16
    if config.MODEL.NAME == 'vit_b':
        config.MODEL.EMBED_DIM = 768
        config.MODEL.DEPTH = 12
        config.MODEL.NUM_HEADS = 12
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16
    elif config.MODEL.NAME == 'vit_l':
        config.MODEL.EMBED_DIM = 1024
        config.MODEL.DEPTH = 24
        config.MODEL.NUM_HEADS = 16
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16
    elif config.MODEL.NAME == 'vit_h':
        config.MODEL.EMBED_DIM = 1280
        config.MODEL.DEPTH = 32
        config.MODEL.NUM_HEADS = 16
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16
    elif config.MODEL.NAME == 'vit_g':
        config.MODEL.EMBED_DIM = 1408
        config.MODEL.DEPTH = 40
        config.MODEL.MLP_RATIO = 48/11
        config.MODEL.NUM_HEADS = 16
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16
    elif config.MODEL.NAME == 'vit_G':
        config.MODEL.EMBED_DIM = 1664
        config.MODEL.DEPTH = 48
        config.MODEL.NUM_HEADS = 16
        config.MODEL.MLP_RATIO = 64/13
        config.MODEL.DECODER_EMBED_DIM = 512
        config.MODEL.DECODER_DEPTH = 8
        config.MODEL.DECODER_NUM_HEADS = 16

    config.freeze()


def get_config(args: Optional[Namespace]):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args is not None:
        update_config(config, args)

    return config
