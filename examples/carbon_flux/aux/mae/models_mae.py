# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import _assert
from timm.models.vision_transformer import Block

from .pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_3d_sincos_pos_embed


Shape3d = Union[List[int], Tuple[int, int, int]]


class PatchEmbed(nn.Module):
    """ 3D PatchEmbed """
    def __init__(
            self,
            input_size: Shape3d = (1, 224, 224),
            patch_size: Shape3d = (1, 16, 16),
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[nn.Module] = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        _assert(H == self.input_size[1], f"Input image height ({H}) doesn't match model ({self.input_size[1]}).")
        _assert(W == self.input_size[2], f"Input image width ({W}) doesn't match model ({self.input_size[2]}).")
        _assert(T == self.input_size[0], f"Input image width ({T}) doesn't match model ({self.input_size[0]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


def generate_mask(x, drop_prob: float = 0., scale_by_keep: bool = True):
    """ Create drop mask for x. Adapted from timm.models.layers.drop_path. """
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return random_tensor


class DropPath(nn.Module):
    """ Adapted from timm.models.layers.DropPath. In this version, drop mask can be saved and reused.
        This is useful when applying the same drop mask more than once.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.drop_mask = None

    def generate_mask(self, x: torch.Tensor):
        self.drop_mask = generate_mask(x, self.drop_prob, self.scale_by_keep)

    def forward(self, x: torch.Tensor, new_mask: bool = True):

        if self.drop_prob == 0. or not self.training:
            return x

        if self.drop_mask is None or new_mask:
            self.generate_mask(x)

        return self.drop_mask * x

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: Optional[int] = None):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        tokens_per_frame: number of tokens for each frame the sample. If provided, embeddings will be
            repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = get_1d_sincos_pos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = get_1d_sincos_pos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = get_1d_sincos_pos_embed_from_grid_torch(
                self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = get_1d_sincos_pos_embed_from_grid_torch(
                self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 input_size: Shape3d = (1, 224, 224),
                 patch_size: Shape3d = (1, 16, 16),
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 coords_encoding: Optional[List[str]] = None,
                 coords_drop_rate: float = 0.0,
                 coords_scale_learn: bool = False,
                 drop_channels_rate: float = 0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.drop_channels = nn.Dropout3d(drop_channels_rate) if drop_channels_rate > 0 else nn.Identity()

        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
            self.drop_temporal = DropPath(coords_drop_rate, scale_by_keep=False)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)
            self.drop_location = DropPath(coords_drop_rate, scale_by_keep=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if self.temporal_encoding:
            self.temporal_embed_dec = TemporalEncoder(decoder_embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_dec = LocationEncoder(decoder_embed_dim, coords_scale_learn)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer("decoder_pos_embed", torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      patch_size[0] * patch_size[1] * patch_size[2] * in_chans,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: B, C, T, H, W
        x: B, L, D
        """
        s, p, q = self.patch_embed.patch_size
        x = rearrange(imgs, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)', s=s, p=p, q=q)

        return x

    def unpatchify(self, x):
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        s, p, q = self.patch_embed.patch_size
        gs = self.patch_embed.grid_size
        imgs = rearrange(x, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)', h=gs[1], w=gs[2], t=gs[0], s=s, p=p, q=q)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor,
                        temporal_coords: Optional[torch.Tensor],
                        location_coords: Optional[torch.Tensor],
                        mask_ratio: float):
        #print('enc received', x.shape)
        # Drop input channels
        x = self.drop_channels(x)
        #print('after drop channel self.drop_channels(x)',x.shape)

        # embed patches
        x = self.patch_embed(x)
        #print('after patch embed self.patch_embed(x)',x.shape)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        #print('after pos embed x + self.pos_embed[:, 1:, :]',x.shape, self.pos_embed[:, 1:, :].shape)

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.input_size[0]
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            temporal_encoding = self.drop_temporal(temporal_encoding, new_mask=True)
            x = x + temporal_encoding
        #print('after temp enc x = x + temporal_encoding', x.shape)
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            location_encoding = self.drop_location(location_encoding, new_mask=True)
            x = x + location_encoding
        #print('after loc enc x = x + location_encoding', x.shape)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        #print('after mask', x.shape, mask.shape, ids_restore.shape)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print('after add class token', x.shape)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #print('after transformer block', x.shape)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor,
                        ids_restore: torch.Tensor,
                        temporal_coords: Optional[torch.Tensor],
                        location_coords: Optional[torch.Tensor]):
        #print('dec received', x.shape)
        # embed tokens
        x = self.decoder_embed(x)
        #print('after dec embed', x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        #print('after app mask token to seq', x.shape)

        # add pos embed
        x = x + self.decoder_pos_embed
        #print('after add pos embed', x.shape)
        # remove cls token
        x_ = x[:, 1:, :]
        #print('after remove class token', x.shape)

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.input_size[0]
            temporal_encoding = self.temporal_embed_dec(temporal_coords, num_tokens_per_frame)
            # Reuse drop mask from encoder for consistent dropping
            temporal_encoding = self.drop_temporal(temporal_encoding, new_mask=False)
            # Add temporal encoding w/o cls token
            x_ = x_ + temporal_encoding
        #print('after temp enc', x.shape)
        if self.location_encoding:
            location_encoding = self.location_embed_dec(location_coords)
            # Reuse drop mask from encoder for consistent dropping
            location_encoding = self.drop_location(location_encoding, new_mask=False)
            # Add location encoding w/o cls token
            x_ = x_ + location_encoding
        #print('after loc enc', x.shape)

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        #print('after append cls token', x.shape)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        #print('after transformer', x.shape)

        # predictor projection
        x = self.decoder_pred(x)
        #print('after decoder proj', x.shape)

        # remove cls token
        x = x[:, 1:, :]
        #print('after remove class token', x.shape)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: B, C, T, H, W
        target: B, L, D
        pred: B, L, D
        mask: B, L. 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, temporal_coords: Optional[torch.Tensor], location_coords: Optional[torch.Tensor], mask_ratio=0.75):

        latent, mask, ids_restore = self.forward_encoder(imgs, temporal_coords, location_coords, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, temporal_coords, location_coords)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=[1, 16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=[1, 16, 16], embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=[1, 14, 14], embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


def build_model(config):

    model = MaskedAutoencoderViT(input_size=config.DATA.INPUT_SIZE,
                                 patch_size=config.MODEL.PATCH_SIZE,
                                 in_chans=len(config.DATA.BANDS),
                                 embed_dim=config.MODEL.EMBED_DIM,
                                 depth=config.MODEL.DEPTH,
                                 num_heads=config.MODEL.NUM_HEADS,
                                 decoder_embed_dim=config.MODEL.DECODER_EMBED_DIM,
                                 decoder_depth=config.MODEL.DECODER_DEPTH,
                                 decoder_num_heads=config.MODEL.DECODER_NUM_HEADS,
                                 mlp_ratio=config.MODEL.MLP_RATIO,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 norm_pix_loss=config.MODEL.NORM_PIX_LOSS,
                                 coords_encoding=config.MODEL.COORDS_ENCODING,
                                 coords_drop_rate=config.MODEL.COORDS_DROP_RATE,
                                 coords_scale_learn=config.MODEL.COORDS_SCALE_LEARN,
                                 drop_channels_rate=config.MODEL.DROP_CHANNELS_RATE)
    return model
