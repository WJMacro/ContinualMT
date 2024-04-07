import torch
import torch.nn as nn
from torch import Tensor
from ..adapter.adapter import Adapter
from .MoA_config import MoAConfig
from fairseq.modules import MultiheadAttention

class MoALayer(nn.Module):
    '''
    Mixture of Adapters layer
    '''
    def __init__(
        self, 
        cfg: MoAConfig, 
        input_dim: int
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.num_adapter = cfg.num_adapter
        self.adapters = nn.ModuleList()
        for _ in range(self.num_adapter):
            self.adapters.append(Adapter(
                cfg.adapter, 
                input_dim
            ))
        self.fusion_type = cfg.fusion_type
        # Gate fusion
        if self.fusion_type == 'gate':
            self.gate = nn.Linear(self.input_dim, self.num_adapter)
        # Attention fusion
        elif self.fusion_type == 'attention':
            self.attention = MultiheadAttention(
                self.input_dim, 
                1, 
                dropout=0.1
            )
    
    def forward(self, x, adapter_id=-1, fusion=False):
        """
        Forward pass for MoA layer.
        Args:
            x (Tensor): input tensor with shape `(seq_len, batch_size, embed_dim)`
            adapter_id (int): forward pass through the `adapter_id`-th adapter
            fusion (Tensor): fusion tensor with shape `(seq_len, batch_size, embed_dim)`

            Note that `adapter_id` and `fusion` cannot be specified at the same time.

        Returns:
            Tensor: output tensor with shape `(seq_len, batch_size, embed_dim)`
        """

        if adapter_id >= 0 and fusion:
            raise ValueError("adapter_id and fusion cannot be specified at the same time.")
        
        # Forward pass through the `adapter_id`-th adapter
        if adapter_id >= 0:
            return self.adapters[adapter_id](x)
        
        # Forward pass through the original transformer
        if adapter_id < 0 and not fusion:
            return x

        # Forward pass through the fusion layer
        if fusion:
            adapter_output = [self.adapters[i](x) for i in range(self.num_adapter)]
            adapter_output = torch.stack(adapter_output, dim=0)
            
            if self.fusion_type == 'add':
                return torch.sum(adapter_output, dim=0)
            
            elif self.fusion_type == 'mean':
                return torch.mean(adapter_output, dim=0)

            elif self.fusion_type == 'gate':
                gate_value = self.gate(x)
                gate_weight = torch.softmax(gate_value, dim=-1).unsqueeze(-1)
                return torch.sum(adapter_output * gate_weight, dim=0)

            elif self.fusion_type == 'attention':
                # compute attention, Q is the input, K and V are the adapter outputs
                fusion_output, _ = self.attention(x, adapter_output, adapter_output)
                return fusion_output
        
