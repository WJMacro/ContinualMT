import re
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from omegaconf import II


@register_criterion(
    "kl_divergence",
    dataclass=LabelSmoothedCrossEntropyCriterionConfig,
)
class KLDivergence(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.train()
        net_output = model(**sample["net_input"])

        model.eval()
        teacher_net_output = model(**sample["net_input"])

        model.train()

        loss = self.compute_loss(
            model, net_output, teacher_net_output, sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if target.size(0) != lprobs.size(0):
            target = torch.cat([target, target.clone()], dim=0)

        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, teacher_net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)

        loss = compute_kl_loss(model, net_output, teacher_net_output, pad_mask)

        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        rdrop_kl_loss = utils.item(
            sum(log.get("rdrop_kl_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2)
        )
        if rdrop_kl_loss > 0:
            metrics.log_scalar("rdrop_kl_loss", rdrop_kl_loss)

def duplicate_input(sample):
    if "net_input" in sample.keys():
        sample_input = sample["net_input"]
    else:
        sample_input = sample

    for k, v in sample_input.items():
        if isinstance(v, torch.Tensor):
            sample_input[k] = torch.cat([v, v.clone()], dim=0)
    if "net_input" in sample.keys():
        sample["net_input"] = sample_input
    else:
        sample = sample_input
    return sample


def compute_kl_loss(model, net_output, teacher_net_output, pad_mask=None, reduce=True):
    net_prob = model.get_normalized_probs(net_output, log_probs=True)
    net_prob_tec = model.get_normalized_probs(teacher_net_output, log_probs=False)

    net_prob = net_prob.view(-1, net_prob.size(-1))
    net_prob_tec = net_prob_tec.view(-1, net_prob_tec.size(-1))

    loss = torch.nn.functional.kl_div(net_prob, net_prob_tec, reduction="none")

    if pad_mask is not None:
        loss = loss.masked_fill(pad_mask, 0)

    if reduce:
        loss = loss.sum()

    return loss
