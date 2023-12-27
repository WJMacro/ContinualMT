
import os
import copy
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
from .kd_translation import KDTranslationConfig


@register_task("rec_translation", dataclass=KDTranslationConfig)
class RecTranslationTask(TranslationTask):

    def __init__(self, cfg: KDTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.warmup_steps = 0
        self.teacher_model = None

    @classmethod
    def setup_task(cls, cfg: KDTranslationConfig, **kwargs):
        task = super().setup_task(cfg, **kwargs)
        
        return task

    def build_model(self, cfg, from_checkpoint=False):
        if os.path.exists(cfg.teacher_model_path):
            print("Loading teacher model from {}".format(cfg.teacher_model_path))
            self.teacher_model = self.build_teacher_model(cfg, from_checkpoint).cuda()
        else:
            print("Teacher model not found at {}".format(cfg.teacher_model_path))
            self.teacher_model = None
        model = super().build_model(cfg, from_checkpoint)
        return model

    def build_teacher_model(self, cfg, from_checkpoint=False):

        teacher_cfg = copy.deepcopy(cfg)
        if getattr(teacher_cfg, "arch", None) is not None:
            model_name = getattr(teacher_cfg, "arch")
            original_model_name = model_name.split("@")[1]
            setattr(teacher_cfg, "arch", original_model_name)
        if getattr(teacher_cfg, "_name", None) is not None:
            model_name = getattr(teacher_cfg, "_name")
            original_model_name = model_name.split("@")[1]
            setattr(teacher_cfg, "_name", original_model_name)
        
        model = super().build_model(teacher_cfg, from_checkpoint)
        model.load_state_dict(torch.load(teacher_cfg.teacher_model_path)['model'], strict=True)
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
                loss, sample_size, logging_output = criterion(model, sample, teacher_model = self.teacher_model)
        if ignore_grad:
            loss *= 0
        loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        return loss, sample_size, logging_output



