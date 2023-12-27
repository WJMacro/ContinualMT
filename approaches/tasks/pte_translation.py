
import re
import os
import math
import torch
import torch.nn.functional as F
from fairseq.optim.amp_optimizer import AMPOptimizer

from dataclasses import dataclass, field
from typing import Optional
from fairseq.tasks import register_task
from fairseq.tasks.translation import(
    TranslationConfig,
    TranslationTask,
)

@dataclass
class pteTranslationConfig(TranslationConfig):
    freeze_mask_path: Optional[str] = field(
        default="",
        metadata={"help": "path to the mask to freeze"}
    )
    tunable_mask_path: Optional[str] = field(
        default="",
        metadata={"help": "path to the mask to tune"}
    )
    enable_knowledge_distillation: bool = field(
        default=False,
        metadata={"help": "enable knowledge distillation"}
    )
    teacher_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path to the teacher model"}
    )


@register_task("pte_translation", dataclass=pteTranslationConfig)
class pteTranslationTask(TranslationTask):


    def __init__(self, cfg: pteTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        
        if os.path.exists(cfg.tunable_mask_path):
            self.tunable_mask = torch.load(cfg.tunable_mask_path)
        else:
            self.tunable_mask = None
            print("No parameter tunable")

        
        if os.path.exists(cfg.freeze_mask_path):
            freeze_mask = torch.load(cfg.freeze_mask_path)
            for key in freeze_mask.keys():
                self.tunable_mask[key] =  (1 - freeze_mask[key]) * self.tunable_mask[key]
        else:
            print("No mask to freeze")

        # Combine the two masks, get the tunable mask
        self.modules = get_module()
        self.enable_knowledge_distillation = cfg.enable_knowledge_distillation

    @classmethod
    def setup_task(cls, cfg: pteTranslationConfig, **kwargs):
        task = super().setup_task(cfg, **kwargs)
        
        return task
    
    def build_model(self, cfg, from_checkpoint=False):
        if self.enable_knowledge_distillation and os.path.exists(cfg.teacher_model_path):
            print("Loading teacher model from {}".format(cfg.teacher_model_path))
            self.teacher_model = self.build_teacher_model(cfg, from_checkpoint).cuda()
        else:
            print("Teacher model not found at {}".format(cfg.teacher_model_path))
            self.teacher_model = None
        model = super().build_model(cfg, from_checkpoint)

        # prune the model according to the tunable mask
        if self.tunable_mask is not None:
            for name, param in model.named_parameters():
                maskname = name.replace("module.", "")
                if maskname in self.modules:
                    param.data.mul_(self.tunable_mask[maskname].cuda())
                else:
                    print("Parameter {} not found in the mask".format(maskname))

        return model

    def build_teacher_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        model.load_state_dict(torch.load(cfg.teacher_model_path)['model'], strict=True)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                if self.enable_knowledge_distillation:
                    loss, sample_size, logging_output = criterion(model, sample, teacher_model = self.teacher_model)
                else:
                    loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        return loss, sample_size, logging_output
    
    def optimizer_step(self, optimizer, model, update_num):
        '''
        Override this method to freeze the parameters
        '''

        # print parameter names

        if self.tunable_mask is not None:
            for name, param in model.named_parameters():
                maskname = name.replace("module.", "")
                if maskname in self.modules:
                    # param.grad.data.masked_fill_(~self.tunable_mask[maskname].byte().cuda(), 0)
                    param.grad.data.mul_(self.tunable_mask[maskname].cuda())
                else:
                    # print("Parameter {} not found in the mask".format(maskname))
                    param.grad.data.zero_()

        optimizer.step()


def get_module():

    res = ["encoder.embed_tokens.weight","decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    p1 = ["encoder.layers.", "decoder.layers."]
    p2 = ["0.", "1.", "2.", "3.", "4.", "5."]
    self_ = ["self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",]

    cross_ = ["encoder_attn.k_proj.weight", "encoder_attn.k_proj.bias",
            "encoder_attn.v_proj.weight", "encoder_attn.v_proj.bias",
            "encoder_attn.q_proj.weight", "encoder_attn.q_proj.bias",
            "encoder_attn.out_proj.weight", "encoder_attn.out_proj.bias",
            "encoder_attn_layer_norm.weight", "encoder_attn_layer_norm.bias",]
    
    fc_ = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", 
                "final_layer_norm.weight", "final_layer_norm.bias"]
    
    prefix = "module.module."
    for a in p1:
        if a == p1[0]:
            for b in p2[:]:
                for c in self_[:-2]:
                    res.append(a + b + c)
                for c in fc_[:-2]:
                    res.append(a + b + c)
        else:
            for b in p2[:]:
                for c in self_[:-2]:
                    res.append(a + b + c)
                for c in cross_[:-2]:
                    res.append(a + b + c)
                for c in fc_[:-2]:
                    res.append(a + b + c)
    return res
                    
                    

