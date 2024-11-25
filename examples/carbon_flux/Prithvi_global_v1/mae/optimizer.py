import timm.optim.optim_factory as optim_factory
import torch
from yacs.config import CfgNode


def build_optimizer(config: CfgNode, model_without_ddp: torch.nn.Module):
    """ Build optimizer setting weight decay as zero for bias and norm layers """

    param_groups = optim_factory.param_groups_weight_decay(model=model_without_ddp,
                                                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups,
                                  lr=config.TRAIN.LR,
                                  betas=config.TRAIN.OPTIMIZER.BETAS,
                                  eps=config.TRAIN.OPTIMIZER.EPS)

    return optimizer

