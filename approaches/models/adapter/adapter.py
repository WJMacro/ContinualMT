from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.modules import LayerNorm

class Adapter(nn.Module):
    """Adapter block.

    Args:
        cfg (FairseqDataclass): config
    """

    def __init__(self, cfg, input_dim, normalize_before=True):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.bottelneck_dim = cfg.bottelneck_dim
        self.downsample = nn.Linear(self.input_dim, self.bottelneck_dim)
        self.activation = nn.ReLU()
        self.upsample = nn.Linear(self.bottelneck_dim, self.input_dim)
        self.normalize_before = normalize_before
        self.enable_layer_norm = cfg.enable_layer_norm
        if self.enable_layer_norm:
            self.layer_norm = LayerNorm(self.input_dim)


    def forward(self, x):
        """Forward pass for adapter.

        Args:
            x (Tensor): input tensor with shape `(seq_len, batch_size, embed_dim)`
            task_id (int): task id

        Returns:
            Tensor: output tensor with shape `(seq_len, batch_size, embed_dim)`
        """

        if self.normalize_before and self.enable_layer_norm:
            x = self.layer_norm(x)
        
        # residual
        residual = x

        # downsample
        x = self.downsample(x)

        # activation
        x = self.activation(x)

        # upsample
        x = self.upsample(x)

        if self.enable_layer_norm and not self.normalize_before:
            x = self.layer_norm(x)

        # add residual
        x = x + residual



        return x


