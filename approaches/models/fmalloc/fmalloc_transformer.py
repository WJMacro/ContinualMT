# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import checkpoint_utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerModelBase,
    TransformerModel,
)

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.dataclass.utils import gen_parser_from_dataclass

from .fmalloc_transformer_layer import FMALLOCTransformerEncoderLayer, FMALLOCTransformerDecoderLayer
from .fmalloc_transformer_config import FMALLOCTransformerConfig


class FMALLOCTransformerEncoder(TransformerEncoderBase):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`FMALLOCTransformerEncoderLayer`.

    Args:
        cfg (TransformerConfig): transformer config
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        return_fc (bool, optional): return final fc layer output (default: False)
    """

    def __init__(
        self, 
        cfg, 
        dictionary, 
        embed_tokens, 
        return_fc=False
    ):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)
    
    def build_encoder_layer(self, cfg):

        layer = FMALLOCTransformerEncoderLayer(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_previous_task_mask(self):
        prevous_mask = {}
        for id, layer in enumerate(self.layers):
            tmp = layer.get_previous_task_mask()
            for k, v in tmp.items():
                k_ = 'layers.{}.{}'.format(id, k)
                prevous_mask[k_] = v
        
        return prevous_mask

    def get_task_mask(self):
        task_mask = {}
        for id, layer in enumerate(self.layers):
            tmp = layer.get_task_mask()
            for k, v in tmp.items():
                k_ = 'layers.{}.{}'.format(id, k)
                task_mask[k_] = v
        
        return task_mask


class FMALLOCTransformerDecoder(TransformerDecoderBase):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`FMALLOCTransformerDecoderLayer`.

    Args:
        cfg (TransformerConfig): transformer config
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): if set, only self-attention in
            decoder layers (default: False).
        output_projection (torch.nn.Linear, optional): output projection layer
    """

    def __init__(
        self, 
        cfg, 
        dictionary, 
        embed_tokens, 
        no_encoder_attn=False, 
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = FMALLOCTransformerDecoderLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_previous_task_mask(self):
        prevous_mask = {}
        for id, layer in enumerate(self.layers):
            tmp = layer.get_previous_task_mask()
            for k, v in tmp.items():
                k_ = 'layers.{}.{}'.format(id, k)
                prevous_mask[k_] = v
        
        return prevous_mask
    
    def get_task_mask(self):
        task_mask = {}
        for id, layer in enumerate(self.layers):
            tmp = layer.get_task_mask()
            for k, v in tmp.items():
                k_ = 'layers.{}.{}'.format(id, k)
                task_mask[k_] = v
        
        return task_mask



@register_model("fmalloc_transformer")
class FMALLOCTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.supports_align_args = True
        self.previous_mask = None
        

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, FMALLOCTransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # cfg must be a FMALLOCTransformerConfig instance
        cfg = FMALLOCTransformerConfig.from_namespace(args)

        model = super().build_model(cfg, task)

        # load pretrained model
        if not os.path.exists(cfg.pretrained_transformer_path):
            print("Pretrained model not found: {}".format(cfg.pretrained_transformer_path))
        else:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.pretrained_transformer_path)
            pretrained_state_dict = state['model']
            model.load_state_dict(pretrained_state_dict, strict=False)
            print("Load pretrained model from {}".format(cfg.pretrained_transformer_path))

        if cfg.enable_hat:
            # freeze pretrained model
            for name, param in model.named_parameters():
                if 'fc1' in name or 'fc2.weight' in name:
                    continue
                if 'task_embedding' in name:
                    continue
                param.requires_grad = False

            # load general domain mask
            if not os.path.exists(cfg.general_domain_mask_path):
                print("General domain mask not found: {}".format(cfg.general_domain_mask_path))
            else:

                mask = torch.load(cfg.general_domain_mask_path)
                task_num = cfg.hat.task_num
                for key in mask.keys():
                    name = key.replace('ffn_mask', 'ffn_hat.task_embedding.weight')                    
                    quantile = mask[key].flatten().quantile(cfg.sparsity) 
                    model.state_dict()[name][0] = (mask[key] > quantile) + (-1) * (mask[key] <= quantile)
                    model.state_dict()[name][0] = model.state_dict()[name][0].to(torch.float) * cfg.hat.thres_emb
                    for i in range(1, model.state_dict()[name].shape[0]):
                        model.state_dict()[name][i] += (mask[key] > quantile) * cfg.hat.thres_emb

        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
             FMALLOCTransformerConfig.from_namespace(args), 
             dictionary, 
             embed_dim, 
             path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FMALLOCTransformerEncoder(             
            FMALLOCTransformerConfig.from_namespace(args), 
            src_dict, 
            embed_tokens
        )
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return FMALLOCTransformerDecoder(             
            FMALLOCTransformerConfig.from_namespace(args), 
            tgt_dict,
            embed_tokens,
        )
    
    def get_previous_task_mask(self):

        if self.previous_mask is not None:
            return self.previous_mask

        self.previous_mask = {}
        encoder_mask = self.encoder.get_previous_task_mask()
        decoder_mask = self.decoder.get_previous_task_mask()

        for k, v in encoder_mask.items():
            k_ = 'encoder.{}'.format(k)
            self.previous_mask[k_] = 1 - v
                
        for k, v in decoder_mask.items():
            k_ = 'decoder.{}'.format(k)
            self.previous_mask[k_] = 1 - v
        
        for k, v in self.previous_mask.items():
            sum = torch.sum(v > 0)
            print("{}:{}".format(k,sum))

        return self.previous_mask
    
    def get_task_mask(self):
        
        task_mask = {}
        encoder_mask = self.encoder.get_task_mask()
        decoder_mask = self.decoder.get_task_mask()

        for k, v in encoder_mask.items():
            k_ = 'encoder.{}'.format(k)
            task_mask[k_] = v
            
        for k, v in decoder_mask.items():
            k_ = 'decoder.{}'.format(k)
            task_mask[k_] = v

        return task_mask


