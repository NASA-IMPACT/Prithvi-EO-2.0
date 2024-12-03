import glob
import os
from functools import partial
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
import xarray as xr
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from torchvision.transforms.functional import resized_crop


HLS_MIN = 0.0
HLS_MAX = 10000.0

SCALING_TYPES = ['none', 'standard', 'norm_clip', 'log1p']


class HLSDataset(Dataset):

    def __init__(self, data: xr.Dataset, bands: List[str], mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None, shape: Optional[List[int]] = None,
                 scaling: str = 'standard', data_augmentation=False, chunk_size: int = 1):
        """ HLSDataset class to load preprocessed zarr files.

        Args:
            data: (xr.Dataset) loaded xarray Dataset.
            bands: list containing the band names to load.
            mean: list containing the mean values for each band (same length as *bands*); ignored when scaling = 'none'.
            std: list containing the std values for each band (same length as *bands*); ignored when scaling = 'none'.
            shape: (list) final shape of the data (limit to this size). If 2D shape is given,
                assumes size 1 in temporal dimension.
            scaling: (str) one of 'standard', 'norm_clip' or 'none'.
            data_augmentation: whether to apply data augmentation transforms (RandomCrop + RandomHorizontalFlip).
            chunk_size: (int) size of the slice to get each batch.
        """
        self.data = data
        self.num_examples = data.sizes['sample']
        self.data_shape = [data.sizes['time'], data.sizes['y'], data.sizes['x']]

        if shape is None:
            shape = self.data_shape
        assert type(shape) in [tuple, list] and len(shape) in [2, 3], f"Incorrect shape format {shape}"
        if len(shape) == 2:   # Handle 2D shape
            shape = [1] + shape

        if not all([shape[i] <= self.data_shape[i] for i in range(3)]):
            raise ValueError(f"shape {shape} incompatible with data {self.data_shape}.")
        if not set(bands).issubset(set(data.band.values.tolist())):
            raise KeyError(f"Not all requested bands {bands} are present in the dataset.")

        if scaling not in SCALING_TYPES:
            raise ValueError(f"Invalid scaling {scaling}. Please select one of {SCALING_TYPES}.")
        if scaling not in ['none', 'log1p'] and (mean is None or std is None):
            raise ValueError(f"Please provide valid mean and std values for {scaling} type.")

        self.shape = shape
        self.bands = bands
        self.data_augmentation = data_augmentation
        self.mean = mean
        self.std = std
        self.scaling = scaling
        self.chunk_size = chunk_size
        self.transform = self._get_transforms()

    @property
    def num_chunks(self):
        return (self.num_examples - self.chunk_size) // self.chunk_size + 1

    def _get_transforms(self):

        transforms = []
        # Select frames if required number < number of frames in self.data
        if self.shape[0] < self.data_shape[0]:
            transforms.append(FrameSelection(num_frames=self.shape[0], random=self.data_augmentation))

        if self.data_augmentation:
            transforms.extend([
                T.RandomCrop(self.shape[-2:]),
                # RandomResizedCrop(self.shape[-2:], scale=(0.67, 1.), ratio=(3./4., 4./3.), antialias=False),
                T.RandomHorizontalFlip()])
        else:
            transforms.extend([T.CenterCrop(self.shape[-2:])])

        if self.scaling == 'standard':
            transforms.append(Normalize(mean=self.mean, std=self.std))
        elif self.scaling == 'norm_clip':
            transforms.append(NormalizeClip(mean=self.mean, std=self.std))
        elif self.scaling == 'log1p':
            transforms.append(Log1pScaling(scale=HLS_MAX))

        return T.Compose(transforms)

    def _get_samples(self, index) -> np.ndarray:
        def get_slice():
            return self.data['bands'].sel(band=self.bands).isel(
                sample=slice(index * self.chunk_size, index * self.chunk_size + self.chunk_size))

        values = np.concatenate([get_slice()], axis=0, dtype=np.float32)   # channels first

        return values

    def _get_date_time(self, index) -> np.ndarray:
        dates = self.data["time_"].isel(
            sample=slice(index * self.chunk_size, index * self.chunk_size + self.chunk_size))

        return self._parse_dates(dates)

    def _get_latlon(self, index) -> np.ndarray:
        latlon = self.data[["center_lat", "center_lon"]].isel(
            sample=slice(index * self.chunk_size, index * self.chunk_size + self.chunk_size))

        return np.stack([latlon["center_lat"], latlon["center_lon"]], axis=-1, dtype=np.float32)

    def preprocess(self, sample: np.ndarray) -> torch.Tensor:
        # Clip values to acceptable range
        sample = torch.from_numpy(sample)
        sample = torch.clip(sample, min=HLS_MIN, max=HLS_MAX)

        sample = self.transform(sample).contiguous()

        return sample

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index):
        sample = self._get_samples(index)
        sample = self.preprocess(sample)
        date_time = torch.from_numpy(self._get_date_time(index))
        latlon = torch.from_numpy(self._get_latlon(index))

        return {"sample": sample, "temporal_coords": date_time, "location_coords": latlon}

    def _parse_dates(self, dates: xr.DataArray):
        years = dates.dt.year
        days = dates.dt.dayofyear - 1

        return np.stack([years, days], axis=-1, dtype=np.float32)

    def __repr__(self):
        return f"{self.__class__.__name__}(examples: {self.num_examples}, " \
               f"chunks: {len(self)}, shape: sample - {self.shape}; temporal_coords - 2)"


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """ Adapted from https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py#L93

    Sampler that can start from a predefined iteration. This is useful when one epoch takes too
    long to finish, and we want to save checkpoints and resume training in the middle of an epoch.
    Notes:
         - Each replica receives a unique set of indices to handle and shuffles *only* the data that
           replica is supposed to view.
         - To continue training from a predefined iteration, use set_start_iter(start_iter) before
           the epoch loop start. Then, set it back to zero if user wants to continue the other
           epochs without skipping the first iterations.
         - We have drop_last behavior by default (if num_replicas > 1); uneven divisions are not allowed.
         - Shuffle is also enabled by default and cannot be removed.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle.
            batch_size (int): batch size that will be used with DataLoader so that indices are
                properly selected when start_iter > 0.
            num_replicas (int, optional): Number of processes participating in distributed training.
                By default, :attr:`world_size` is retrieved from the current distributed group.
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed group.
            seed (int): Seed for the torch generator.
        """
        # Shuffle is controlled here, so set it to False in parent class
        super().__init__(dataset, shuffle=False, seed=seed, num_replicas=num_replicas, rank=rank)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        # partition data into num_replicas and shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = torch.arange(self.rank * self.num_samples, (self.rank + 1) * self.num_samples)[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter


class RandomResizedCrop(T.RandomResizedCrop):
    """ RandomResizeCrop on H, W for data with shape (B, ..., H, W). """

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        shape = img.size()
        img = img.reshape(shape[0], -1, shape[-2], shape[-1])   # flatten channels + T if present
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        img = img.reshape(shape)

        return img


class FrameSelection(torch.nn.Module):
    """ Selects frames from a video tensor.

        The tensor is expected to have [..., T, H, W] shape, where ... means an arbitrary number of leading dimensions.
        If random=True, random frames are selected, but they are ordered after.
        If random=False, center frames are selected.

    Args:
        num_frames: (int) number of frames to select.
        random: (bool) If True, the selection is random, otherwise, the center *num_frames* are selected.
    """

    def __init__(self, num_frames: int, random: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.random = random

    @staticmethod
    def get_idx(x: torch.Tensor, size: int, random: bool):
        """ Get list of indices to subsample. """
        t = x.shape[-3]

        if t < size:
            raise ValueError(f"Required size {size} is larger than input dim {t}")

        if random:
            idx, _ = torch.torch.randperm(t)[:size].sort()
            idx = idx.tolist()
        else:
            start = int(round((t - size) / 2.0))
            stop = start + size
            idx = list(range(start, stop))

        return idx

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.tensor with shape (..., T, H, W)

        Returns:
            Tensor with shape (..., num_frames, H, W).
        """

        if x.shape[-3] == self.num_frames:
            return x

        idx = self.get_idx(x, size=self.num_frames, random=self.random)
        x = x[..., idx, :, :]

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_frames={self.num_frames}, random={self.random})"


class Normalize(torch.nn.Module):
    """ Normalize on channels for data with shape (B, C, ..., H, W).
        Based on the original torchvision transform Normalize.

        Given mean: (mean[1], ..., mean[n]) and std: (std[1], ..., std[n]) for N channels, normalize each channel as:
            output[channel] = (input[channel] - mean[channel]) / std[channel]
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        super().__init__()
        self.mean = list(mean)
        self._check_std(std)
        self.std = list(std)
        self.inplace = inplace

    @staticmethod
    def _check_std(std):
        if isinstance(std, (tuple, list)):
            div_zero = not all(std)
        elif isinstance(std, (int, float)):
            div_zero = std == 0
        else:
            div_zero = False
        if div_zero:
            raise ValueError("std evaluated to zero, leading to division by zero.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        dtype = x.dtype
        device = x.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        shape = [-1] + [1] * (x.ndim - 2)

        mean = mean.view(*shape)
        std = std.view(*shape)

        if self.inplace:
            x = x.sub_(mean)
        else:
            x = x.sub(mean)

        return x.div_(std)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(mean={[round(m, 4) for m in self.mean]}"
        format_string += f", std={[round(s, 4) for s in self.std]}"
        return format_string


class NormalizeClip(torch.nn.Module):
    """ Based on https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py#L349 """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.min = [mean[i] - 2 * std[i] for i in range(len(mean))]
        self.max = [mean[i] + 2 * std[i] for i in range(len(mean))]

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        device = x.device
        min_value = torch.as_tensor(self.min, dtype=dtype, device=device)
        max_value = torch.as_tensor(self.max, dtype=dtype, device=device)

        shape = [-1] + [1] * (x.ndim - 2)
        min_value = min_value.view(*shape)
        max_value = max_value.view(*shape)

        x = (x - min_value) / (max_value - min_value)
        x = torch.clip(x, 0.0, 1.0)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(min={[round(m, 4) for m in self.min]}"
        format_string += f", max={[round(s, 4) for s in self.max]}"
        return format_string


class Log1pScaling(torch.nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale should be > 0.")
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = torch.div(x, self.scale)
        x = torch.log1p(x)
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(scale={self.scale})"


def collate_fn(data: List[Dict], batch_size: int):

    keys = list(data[0].keys())
    big_batch = {k: torch.concat([sample[k] for sample in data], dim=0) for k in keys}
    random_idx = torch.randperm(big_batch[keys[0]].shape[0])[:batch_size]

    batch = {k: v[random_idx] for k, v in big_batch.items()}

    return batch


def build_dataloader(cfg, data_dir: str, shuffle: bool, data_aug: bool):
    """ Wrapper to build dataloader for models that use HLSDataset. """

    file_list = sorted(glob.glob(os.path.join(data_dir, 'data*.zarr')))
    ds = xr.concat([xr.open_zarr(file_path, mask_and_scale=False) for file_path in file_list], dim='sample')

    if 'file_id' in ds:
        ds = ds.drop('file_id')

    assert cfg.TRAIN.BATCH_SIZE % cfg.DATA.CHUNK_SIZE == 0  # batch size must be divisible by chunk size!

    dataset = HLSDataset(data=ds, shape=cfg.DATA.INPUT_SIZE, bands=cfg.DATA.BANDS,
                         scaling=cfg.DATA.SCALING, mean=cfg.DATA.MEAN, std=cfg.DATA.STD,
                         data_augmentation=data_aug, chunk_size=cfg.DATA.CHUNK_SIZE)
    if shuffle:
        sampler = StatefulDistributedSampler(dataset, rank=dist.get_rank(), num_replicas=cfg.TRAIN.WORLD_SIZE,
                                             batch_size=cfg.TRAIN.BATCH_SIZE // cfg.DATA.CHUNK_SIZE)
    else:
        sampler = DistributedSampler(dataset, rank=dist.get_rank(), num_replicas=cfg.TRAIN.WORLD_SIZE, shuffle=False)

    collate_function = partial(collate_fn, batch_size=cfg.TRAIN.BATCH_SIZE)

    dataloader = DataLoader(dataset,
                            batch_size=cfg.TRAIN.BATCH_SIZE // cfg.DATA.CHUNK_SIZE,
                            sampler=sampler,
                            num_workers=cfg.DATA.NUM_WORKERS,
                            pin_memory=cfg.DATA.PIN_MEMORY,
                            prefetch_factor=cfg.DATA.PREFETCH_FACTOR,
                            drop_last=True,
                            collate_fn=collate_function)
    return dataloader
