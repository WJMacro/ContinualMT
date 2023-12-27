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

from .adapter_transformer_layer import AdapterTransformerDecoderLayer, AdapterTransformerEncoderLayer
from .adapter_transformer_config import AdapterTransformerConfig


class AdapterTransformerEncoder(TransformerEncoderBase):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`AdapterTransformerEncoderLayer`.

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

        layer = AdapterTransformerEncoderLayer(
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

class AdapterTransformerDecoder(TransformerDecoderBase):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`AdapterTransformerDecoderLayer`.

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
        layer = AdapterTransformerDecoderLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer



@register_model("adapter_transformer")
class AdapterTransformerModel(TransformerModel):
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
            parser, AdapterTransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # cfg must be a AdapterTransformerConfig instance
        cfg = AdapterTransformerConfig.from_namespace(args)

        model = super().build_model(cfg, task)

        # load pretrained model
        if not os.path.exists(cfg.pretrained_transformer_path):
            raise IOError("Model file not found: {}".format(cfg.pretrained_transformer_path))
        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.pretrained_transformer_path)
        pretrained_state_dict = state['model']
        model.load_state_dict(pretrained_state_dict, strict=False)

        # freeze pretrained model
        for name, param in model.named_parameters():
            if name in pretrained_state_dict.keys():
                param.requires_grad = False

        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
             AdapterTransformerConfig.from_namespace(args), 
             dictionary, 
             embed_dim, 
             path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return AdapterTransformerEncoder(             
            AdapterTransformerConfig.from_namespace(args), 
            src_dict, 
            embed_tokens
        )
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AdapterTransformerDecoder(             
            AdapterTransformerConfig.from_namespace(args), 
            tgt_dict,
            embed_tokens,
        )
    
