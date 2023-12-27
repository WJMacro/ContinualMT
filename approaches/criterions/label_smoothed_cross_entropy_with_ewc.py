import re
import os
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
class LabelSmoothedCrossEntropyCriterionWithEWCConfig(
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

    ewc_lambda: float = field(
        default=0.1,
        metadata={"help": "lambda for KD loss"},
    )

    fisher_path: str = field(
        default="",
        metadata={"help": "path to fisher matrix"},
    )



@register_criterion(
    "label_smoothed_cross_entropy_with_ewc",
    dataclass=LabelSmoothedCrossEntropyCriterionWithEWCConfig,
)
class LabelSmoothedCrossEntropyCriterionWithEWC(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ewc_lambda,
        fisher_path,
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
        self.ewc_lambda = ewc_lambda
        if os.path.exists(fisher_path):
            print("Loading fisher matrix from {}".format(fisher_path))
            self.fisher = torch.load(fisher_path)
            for key in self.fisher:
                self.fisher[key] = self.fisher[key].to("cuda")
            del self.fisher['task_id']
        else:
            print("Fisher matrix not found at {}".format(fisher_path))
            self.fisher = None

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

    def forward(self, model, sample, reduce=True, teacher_model=None, fisher=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ewc_loss = torch.tensor(0.0).to("cuda")

        if teacher_model is not None and self.fisher is not None:
            #print('**********Computing EWC loss*************')
            for key in self.fisher:
                teacher_param = teacher_model.get_parameter(key)
                student_param = model.get_parameter(key)
                ewc_loss += torch.sum(
                    self.fisher[key] * (teacher_param - student_param) ** 2
                )
                fisher_sum = torch.sum(self.fisher[key])
                #print('Key: {}, EWC loss: {}, Fisher: {}'.format(key, ewc_loss, fisher_sum))

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        ewc_loss = ewc_loss * sample_size * self.ewc_lambda
        
        loss = loss + ewc_loss
        #print('Loss: {}, EWC loss: {}'.format(loss, ewc_loss))

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ewc_loss": utils.item(ewc_loss.data) if reduce else ewc_loss.data,
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
        ewc_loss_sum = sum(log.get("ewc_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ewc_loss", ewc_loss_sum / ntokens / math.log(2), ntokens, round=3
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
    