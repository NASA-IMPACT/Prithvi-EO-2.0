import datetime
import json
import os
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from timm.scheduler.scheduler import Scheduler


scheduler_type = Union[torch.optim.lr_scheduler.LRScheduler, Scheduler]


def load_checkpoint(ckpt_path: str, model_without_ddp: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, rank: int,
                    scaler: torch.cuda.amp.GradScaler = None, logger=None):
    """ Load model weights and optimizer/scheduler states (if available). """

    start_epoch = 0
    best_val_loss = float("inf")
    start_iter = 0
    last_batch_loss = None

    if os.path.exists(ckpt_path):
        if rank == 0:
            msg = f"Loading checkpoint {ckpt_path}..."
            logger.info(f"{msg}\n") if logger is not None else print(f"\n{msg}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('scheduler') is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])

            if checkpoint.get('scaler') is not None:
                scaler.load_state_dict(checkpoint['scaler'])

            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']

            if checkpoint.get('iter') is not None:
                start_iter = checkpoint['iter'] + 1
                if start_iter == checkpoint['steps_per_epoch']:
                    start_iter = 0      # last epoch complete - roll iteration to the beginning
                else:
                    start_epoch -= 1    # checkpoint epoch incomplete, start at the same epoch
                    last_batch_loss = checkpoint['loss']  # get last avg batch loss to continue epoch

        del checkpoint
        torch.cuda.empty_cache()

    return start_epoch, best_val_loss, start_iter, last_batch_loss


def load_weights(ckpt_path: str, model_without_ddp: torch.nn.Module, rank: int, logger=None):
    """ Load only weights from checkpoint in *ckpt_path*. Partial load is possible (strict=False). """

    if os.path.exists(ckpt_path):
        if rank == 0:
            msg = f"Loading weights from {ckpt_path}..."
            logger.info(f"{msg}\n") if logger is not None else print(f"\n{msg}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # training checkpoints contain more keys, while best validation checkpoints only contain the weights
        load_key = checkpoint['model'] if 'model' in checkpoint else checkpoint
        msg = model_without_ddp.load_state_dict(load_key, strict=False)
        if rank == 0:
            logger.info(f"\n{msg}\n") if logger is not None else print(f"\n{msg}\n")
    return


def distributed_setup(use_gpu: bool):

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if use_gpu:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        local_device = local_rank
    else:
        dist.init_process_group("gloo")
        local_device = 'cpu'

    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    return rank, local_device, world_size


def all_reduce(x: float, device=None):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    x_reduce = torch.tensor(x, device=device)
    if world_size > 1:
        dist.all_reduce(x_reduce, op=dist.ReduceOp.SUM)
        x_reduce /= world_size
    return x_reduce.item()


def get_mean_std(bands: List[str], meta_file_path: str):
    """ Load pre-computed mean and std from meta file for each band in *bands*. """

    # Get input metadata
    with open(meta_file_path, 'r') as f:
        input_meta_data = json.load(f)

    # Handle old keys "image_mean" and "image_standard_deviation"
    mean_key = 'mean' if input_meta_data.get('mean') is not None else 'image_mean'
    std_key = 'std' if input_meta_data.get('std') is not None else 'image_standard_deviation'

    mean, std = [], []
    for b in bands:
        idx = input_meta_data['bands'].index(b)
        mean.append(input_meta_data[mean_key][idx])
        std.append(input_meta_data[std_key][idx])

    return mean, std


def save_checkpoint(ckpt_file_path: str, epoch: int, model_without_ddp: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler: scheduler_type,
                    train_loss: float, best_val_loss: float, num_steps: int, iteration: Optional[int] = None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None):

    # hack to fix OneCycleLR serialization bug
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler_dict = {k: v for k, v in scheduler.state_dict().items() if k != 'anneal_func'}
    else:
        scheduler_dict = scheduler.state_dict()

    torch.save({'epoch': epoch,
                'iter': iteration,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_dict,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'loss': train_loss,
                'best_val_loss': best_val_loss,
                'steps_per_epoch': num_steps},
               ckpt_file_path)


def get_grad_norm(parameters, norm_type=2):
    """ Compute Gradient norm.
        Based on https://github.com/microsoft/Swin-Transformer/blob/main/utils_simmim.py#L68
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.detach().norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm
