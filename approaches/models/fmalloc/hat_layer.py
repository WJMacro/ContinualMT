from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from scipy.stats import norm
from math import sqrt
from fairseq.modules import GradMultiply    

class HATLayer(nn.Module):
    """HAT Layer block.

    Args:
        cfg (FairseqDataclass): HAT config
    """

    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.task_embedding = negative_embedding(cfg.hat.task_num, embed_dim)
        self.activation_gate = nn.Sigmoid()
        self.dummy_param = nn.Parameter(torch.empty(0))
              

    def mask(self, temperature=None, task_id=None):
        """Compute the mask for HAT layer.
        The mask is a sigmoid function of task embedding.

        Returns:
            Tensor: mask tensor with shape `(1, embed_dim)`
        """

        device = self.dummy_param.device
        
        #return torch.ones(self.embed_dim).data.detach().to(device)


        if task_id is None:
            task_id = self.cfg.hat.task_id
        if temperature is None:
            temperature = self.cfg.hat.temperature

        embedding = self.task_embedding(torch.LongTensor([task_id]).to(device))

        if self.training:
            mask = self.activation_gate(temperature*embedding)
        else:
            mask = self.activation_gate(self.cfg.hat.temperature_max*embedding)

        return mask


    def get_previous_task_mask(self):
        """Compute the mask for all previous tasks.

        Returns:
            Tensor: mask tensor with shape `(1, embed_dim)`
        """
        device = self.dummy_param.device   

        previous_mask = torch.zeros(self.embed_dim).to(device)
        
        # if task_id is not 0, return a mask that combines all previous tasks' mask
        for i in range(self.cfg.hat.task_id):
            mask = self.mask(temperature=self.cfg.hat.temperature_max, task_id=i)
            previous_mask = torch.max(previous_mask, mask)

        previous_mask = (previous_mask > 0.5).to(torch.float)
        # print(previous_mask)

        return previous_mask.data.detach().to(device)

    def forward(self, x, task_id=None):
        """Compute the HAT layer.

        Args:
            x (Tensor): input tensor with shape `(seq_len, batch, embed_dim)`

        Returns:
            Tensor: output tensor with shape `(seq_len, batch, embed_dim)`
        """
        # (seq_len, batch, embed_dim)
        mask = self.mask(temperature=self.cfg.hat.temperature, task_id=task_id)
        #print(mask)
        x = x * mask

        return x


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    for idx in range(num_embeddings):
        mean = norm.ppf(1 / (num_embeddings - idx) * 0.9999)
        nn.init.normal_(m.weight.data[idx], mean=mean, std=1)
        # normalize embedding
        m.weight.data[idx] = m.weight.data[idx] / m.weight.data[idx].norm(2, -1, keepdim=True)
        
    return m

def negative_embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    # generate normal distribution, make all embeddings negative
    for idx in range(num_embeddings):
        m.weight.data[idx] = torch.abs(m.weight.data[idx]) * -1
        
    return m

if __name__ == "__main__":
    from approaches.models.hat.hat_transformer_ffn_config import HATTransformerConfig

    cfg = HATTransformerConfig()
    cfg.hat.task_num = 5
    cfg.hat.temperature = 0.1
    cfg.hat.temperature_max = 10
    cfg.hat.thres_cosh = 2
    cfg.hat.thres_emb = 0.5
    cfg.hat.task_id = 0
    embed_dim = 16
    hat_layer = HATLayer(cfg, embed_dim).to("cuda")
    x = torch.rand(32, 16, embed_dim).to("cuda")
    mask = hat_layer.mask()

    print(mask)
    loss = torch.sum(mask)
    print(loss)

    loss.backward()

    for name, param in hat_layer.named_parameters():
        print(name, param.grad)
