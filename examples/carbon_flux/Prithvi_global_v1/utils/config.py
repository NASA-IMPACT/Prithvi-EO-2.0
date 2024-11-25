from argparse import Namespace
from yacs.config import CfgNode as CN


def update_value(config: CN, key: str, value):
    config.defrost()
    config.merge_from_list([key, value])
    config.freeze()


def check_args(name: str, args: Namespace):
    if hasattr(args, name) and eval(f'args.{name}') is not None:
        return True
    return False


def update_common_args(args: Namespace, config: CN):
    """ Helper to update command line args that are common to all models. """

    # Update additional list of options, if available
    if args.opts is not None:
        config.merge_from_list(args.opts)

    # Update specific params from command line (data and training details)
    if check_args('train_dir', args):
        config.DATA.TRAIN_DIR = args.train_dir
    if check_args('val_dir', args):
        config.DATA.VAL_DIR = args.val_dir
    if check_args('num_workers', args):
        config.DATA.NUM_WORKERS = args.num_workers
    if check_args('meta_file_path', args):
        config.DATA.META_FILE_PATH = args.meta_file_path
    if check_args('mask_ratio', args):
        config.DATA.MASK_RATIO = args.mask_ratio
    if check_args('data_augmentation', args):
        config.DATA.DATA_AUGMENTATION = args.data_augmentation

    if check_args('batch_size', args):
        config.TRAIN.BATCH_SIZE = args.batch_size
    if check_args('epochs', args):
        config.TRAIN.EPOCHS = args.epochs
    if check_args('warmup_epochs', args):
        config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    if check_args('lr', args):
        config.TRAIN.LR = args.lr
    if check_args('resume', args):
        config.TRAIN.RESUME = args.resume
    if check_args('load_weights', args):
        config.TRAIN.LOAD_WEIGHTS = args.load_weights
    if check_args('base_output_dir', args):
        config.TRAIN.BASE_OUTPUT_DIR = args.base_output_dir
    if check_args('cpu_only', args):
        config.TRAIN.CPU_ONLY = args.cpu_only

    return

