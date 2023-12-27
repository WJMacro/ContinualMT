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


@register_criterion(
    "label_smoothed_cross_entropy_with_capacity",
    dataclass=LabelSmoothedCrossEntropyCriterionConfig,
)
class LabelSmoothedCrossEntropyCriterionWithCapacity(LabelSmoothedCrossEntropyCriterion):
    
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
            ignore_prefix_size,
            report_accuracy,
        )

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # Capacity usage
        
        previous_mask = model.get_previous_task_mask()
        current_mask = model.get_task_mask()
        used_mask = {}


        aux_count = 0
        new_count = 0
        for key in (previous_mask.keys() & current_mask.keys()):
            aux_count += previous_mask[key].numel()
            new_count += (previous_mask[key] * current_mask[key]).sum()


        capacity_usage = new_count / aux_count


        if reduce:
            capacity_usage = capacity_usage.sum()
        

        return loss, nll_loss, capacity_usage

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, capacity_usage = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        capacity_usage = capacity_usage * sample_size / model.cfg.sparsity


        # print("loss: {}, nll_loss: {}, capacity_usage: {}, lambda: {}".format(loss, nll_loss, capacity_usage, self.lambd))

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "capacity_usage": utils.item(capacity_usage.data) if reduce else capacity_usage.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "temperature": model.cfg.hat.temperature,
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
        capacity_usage_sum = sum(log.get("capacity_usage", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        temperature = logging_outputs[-1].get("temperature", 1)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "capacity_usage", capacity_usage_sum / sample_size * 100 , sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "temperature", temperature, 1, round=3
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
    