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

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,use_cls=True, **kwargs): #
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.patch_embed = PatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                      in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim']) # changed - added
        self.use_cls = use_cls # use cls token or not
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            
            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # changed - added in_chans

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            outcome = x[:, 1:, :].mean(dim=1).unsqueeze(1)  # global pool without cls token
            # FIXME: Use only for global pool = True
            outcome = self.fc_norm(x)
        elif self.use_cls:
            x = self.norm(x)
            outcome = x[:, 0]
        else:
            x = self.norm(x)
            outcome = x[:,1:] # [B, nvar*num_win, embed_dim]

        return outcome

    # FIXME: relax this constraint: only used for global pool
    # def forward_head(self,x):
    #     print('x shape before fc_norm:', x.shape)
    #     x = self.fc_norm(x)

    #     return x


def vit_base_patch16(use_cls=True,**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),use_cls=use_cls, **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



#def mae_vit_tiny_patch16_dec256d8b(**kwargs): for vit_tiny_patch16
#    model = MaskedAutoencoderViT(
#        embed_dim=192, depth=12, num_heads=3, 
#        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model
