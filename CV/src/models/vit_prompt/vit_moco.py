#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_moco import VisionTransformerMoCo
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class PromptedVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        if self.prompt_config.DEEP:
            self.prompt_depth = len(self.blocks)
        else:
            self.prompt_depth = 1

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((
                cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        if self.prompt_config.KV_ONLY:
            x = self.embeddings(x)

            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    prompt = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
                    x = self.blocks[i](x, prompt=prompt)
                elif i < self.prompt_depth and self.prompt_config.DEEP:
                    prompt = self.prompt_dropout(self.deep_prompt_embeddings[i-1].expand(B, -1, -1))
                    x = self.blocks[i](x, prompt=prompt)
                else:
                    x = self.blocks[i](x)
        else:
            x = self.incorporate_prompt(x)

            if self.prompt_config.DEEP:
                B = x.shape[0]
                num_layers = len(self.blocks)

                for i in range(num_layers):
                    if i == 0:
                        x = self.blocks[i](x)
                    else:
                        # prepend
                        x = torch.cat((
                            x[:, :1, :],
                            self.prompt_dropout(
                                self.deep_prompt_embeddings[i-1].expand(B, -1, -1)
                            ),
                            x[:, (1 + self.num_tokens):, :]
                        ), dim=1)
                        x = self.blocks[i](x)
            else:
                for blk in self.blocks:
                    x = blk(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        



class PrefixVisionTransformerMoCo(VisionTransformerMoCo):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.num_experts = num_tokens // 2

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        if self.prompt_config.DEEP:
            self.prompt_depth = len(self.blocks)
        else:
            self.prompt_depth = 1

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            if prompt_config.SHARE_KV:
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    self.prompt_depth, self.num_experts, self.embed_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                if prompt_config.SHARE_TYPE == "non-linear":
                    self.prompt_trans = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.embed_dim, 48),
                            nn.Tanh(),
                            nn.Linear(48, 2 * self.embed_dim)
                        ) for _ in range(self.prompt_depth)
                    ])
                else:
                    self.prompt_trans = nn.ModuleList([
                        nn.Linear(self.embed_dim, 2 * self.embed_dim) for _ in range(self.prompt_depth)
                    ])
            else:
                self.prompt_key_embeddings = nn.Parameter(torch.zeros(
                    self.prompt_depth, self.num_experts, self.embed_dim))
                self.prompt_value_embeddings = nn.Parameter(torch.zeros(
                    self.prompt_depth, self.num_experts, self.embed_dim))
                
                nn.init.uniform_(self.prompt_key_embeddings.data, -val, val)
                nn.init.uniform_(self.prompt_value_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")


    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((
                cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
            if self.prompt_config.SHARE_KV:
                self.prompt_trans.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.embeddings(x)

        num_layers = len(self.blocks)
        B = x.shape[0]

        for i in range(num_layers):
            if i == 0 or (i < self.prompt_depth and self.prompt_config.DEEP):
                if self.prompt_config.SHARE_KV:
                    prefix_key_value = self.prompt_trans[i](
                        self.prompt_embeddings[i]
                    ).reshape(self.num_experts, -1, 2).permute(2, 0, 1)

                    prefix_key = self.prompt_dropout(prefix_key_value[0].expand(B, -1, -1))
                    prefix_value = self.prompt_dropout(prefix_key_value[1].expand(B, -1, -1))

                else:
                    prefix_key = self.prompt_dropout(
                        self.prompt_key_embeddings[i].expand(B, -1, -1))
                    prefix_value = self.prompt_dropout(
                        self.prompt_value_embeddings[i].expand(B, -1, -1))
                    
                x = self.blocks[i](x, prefix_key=prefix_key, prefix_value=prefix_value)
            else:
                x = self.blocks[i](x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]



def vit_base(prompt_cfg, **kwargs):
    if prompt_cfg.PREFIX_TUNING:
        model = PrefixVisionTransformerMoCo(
            prompt_cfg,
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    else:
        model = PromptedVisionTransformerMoCo(
            prompt_cfg,
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
    model.default_cfg = _cfg()
    return model

