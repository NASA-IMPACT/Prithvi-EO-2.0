import glob
import os
from functools import partial
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from yacs.config import CfgNode

import utils.data as ut
from swin.data import MaskGenerator, HLSDatasetMask


def build_data_loader(data_dir: str, cfg: CfgNode, num_samples: int, mask_ratio: Optional[float], mask_gen=True):
    """ Wrapper to build data loader for inference. """

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    ds = xr.concat([xr.open_zarr(file_path, mask_and_scale=False) for file_path in file_list], dim='sample')
    if 'file_id' in ds:
        ds = ds.drop('file_id')

    # Build dataset with chunk size = 1 for random selection
    if mask_gen:
        mask_generator = MaskGenerator(input_size=cfg.DATA.INPUT_SIZE,
                                       mask_patch_size=cfg.DATA.MASK_PATCH_SIZE,
                                       model_patch_size=cfg.MODEL.PATCH_SIZE,
                                       mask_ratio=mask_ratio)

        dataset = HLSDatasetMask(data=ds, shape=cfg.DATA.INPUT_SIZE, bands=cfg.DATA.BANDS,
                                 scaling=cfg.DATA.SCALING, mean=cfg.DATA.MEAN, std=cfg.DATA.STD,
                                 data_augmentation=False, chunk_size=1, mask_generator=mask_generator)
    else:
        dataset = ut.HLSDataset(data=ds, shape=cfg.DATA.INPUT_SIZE, bands=cfg.DATA.BANDS,
                                scaling=cfg.DATA.SCALING, mean=cfg.DATA.MEAN, std=cfg.DATA.STD,
                                data_augmentation=False, chunk_size=1)

    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples)
    collate_function = partial(ut.collate_fn, batch_size=1)

    print(f"Dataset length = {dataset.num_examples:,}")

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,  # = 1 to compute losses for each sample separately
                                         sampler=sampler,
                                         drop_last=False,
                                         collate_fn=collate_function)

    return loader


def plot_predictions(imgs: dict[str, torch.Tensor], losses, bands: List[str],
                     output_dir: str, loss_name: str, fig_size=(16, 12), cmap=plt.cm.viridis):
    """ Plot results for each band and save in *output_dir*.

    Args:
        imgs: dict with results in the format: {key: tensor with shape (sample, C, T, H, W)};
            keys are: ['original', 'predicted', 'mask', 'predicted_full'], as returned by run function.
        losses: list with the losses for each mask ratio.
        bands: list with name of the bands.
        output_dir: (str) path to the directory where to save plots.
        loss_name: (str) name of the loss function, to appear in the plot.
        fig_size: tuple containing the figure size in (W, H).
        cmap: matplotlib colormap for plots.
    """

    num_timesteps = imgs['original'].shape[2]

    for sample in range(len(losses)):

        for band, name in enumerate(bands):

            fig, ax = plt.subplots(nrows=num_timesteps, ncols=4, figsize=fig_size, constrained_layout=True)
            if len(ax.shape) == 1:
                ax = ax[np.newaxis, :]

            for t in range(num_timesteps):

                original = imgs['original'][sample, band, t, :, :]
                pred = imgs['predicted'][sample, band, t, :, :]
                pred_full = imgs['predicted_full'][sample, band, t, :, :]
                mask = imgs['mask'][sample, band, t, :, :].clone()

                mask[mask == 1.0] = np.nan  # only masked regions appear in black; the rest is transparent

                # Ensuring same color scale for all images in a row
                vmin = original.min()
                vmax = original.max()

                if t == 0:
                    ax[t, 0].set_title(f'original - {name}', fontsize=10)
                    ax[t, 1].set_title('mask', fontsize=10)
                    ax[t, 2].set_title('predicted', fontsize=10)
                    ax[t, 3].set_title(f"visible + predicted , {loss_name} = {losses[sample]:.4f}", fontsize=10)

                if num_timesteps > 1:
                    ax[t, 0].set_ylabel(f't{t}', fontsize=10)

                ax[t, 0].imshow(original, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[t, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

                ax[t, 1].imshow(original, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[t, 1].imshow(mask, cmap=plt.cm.gray, alpha=1.0)  # plot mask on top of original image
                ax[t, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

                ax[t, 2].imshow(pred_full, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[t, 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

                ax[t, 3].imshow(pred, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[t, 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            plt.savefig(os.path.join(output_dir, f'result_{sample}_{name}.png'))
