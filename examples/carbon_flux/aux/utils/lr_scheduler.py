import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from yacs.config import CfgNode


def build_scheduler(config: CfgNode, optimizer: torch.optim.Optimizer, n_iter_per_epoch: int):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            cycle_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
        )
    else:
        raise NotImplementedError(f"Specified LR scheduler {config.TRAIN.LR_SCHEDULER.NAME} not implemented")

    return lr_scheduler
