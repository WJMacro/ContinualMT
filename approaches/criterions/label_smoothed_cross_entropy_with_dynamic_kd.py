import re
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionWithDynamicKDConfig(
    FairseqDataclass
):
    
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    kd_lambda: float = field(
        default=0.1,
        metadata={"help": "lambda for KD loss"},
    )



@register_criterion(
    "label_smoothed_cross_entropy_with_dynamic_kd",
    dataclass=LabelSmoothedCrossEntropyCriterionWithDynamicKDConfig,
)
class LabelSmoothedCrossEntropyCriterionWithDynamicKD(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kd_lambda,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
        )
        self.kd_lambda = kd_lambda

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    def forward(self, model, sample, reduce=True, teacher_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = nll_loss

        if teacher_model is not None:
            teacher_net_output = teacher_model(**sample["net_input"])
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            teacher_probs = teacher_model.get_normalized_probs(teacher_net_output)

            lprobs = lprobs.view(-1, lprobs.size(-1))
            teacher_probs = teacher_probs.view(-1, teacher_lprobs.size(-1))

            target = model.get_targets(sample, net_output).view(-1, 1)
            non_pad_mask = target.ne(self.padding_idx)
            
            kd_loss = -(teacher_probs * lprobs).sum(dim=-1, keepdim=True)[non_pad_mask]

            loss = (1 - self.kd_lambda) * loss + self.kd_lambda * kd_loss

        else:
            kd_loss = torch.tensor(0.0)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "kd_loss": utils.item(kd_loss.data) if reduce else kd_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "kd_loss", kd_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
    