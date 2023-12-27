
import re
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



@register_task("fmalloc_translation", dataclass=TranslationConfig)
class FMALLOCTranslationTask(TranslationTask):


    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        task = super().setup_task(cfg, **kwargs)
        
        return task

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        return model
    
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        epoch_itr = super().get_batch_iterator(
            dataset,
            max_tokens,
            max_sentences,
            max_positions,
            ignore_invalid_inputs,
            required_batch_size_multiple,
            seed,
            num_shards,
            shard_id,
            num_workers,
            epoch,
            data_buffer_size,
            disable_iterator_cache,
            skip_remainder_batch,
            grouped_shuffling,
            update_epoch_batch_itr,
        )
        # get the number of batches in the epoch
        num_batches = len(epoch_itr)
        # if its the first epoch and we are using the trainset, we need to set the warmup steps

        if epoch == 1 and "train" in self.datasets.keys() and dataset == self.datasets["train"]:
            self.num_batches = num_batches
        
        return epoch_itr

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
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def optimizer_step(self, optimizer, model, update_num):
        '''
        Override this method to add task mask to the gradient
        '''

        # TODO: this is a hack to get the mask to the optimizer
        # we should probably just pass the closure to the optimizer

        _BLOCK_NAME_RE = re.compile(r"(.*).(fc)(1|2).(weight|bias)$")

        temp = model.cfg.hat.temperature
        temp_min = model.cfg.hat.temperature_min
        temp_max = model.cfg.hat.temperature_max
        thres_cosh = model.cfg.hat.thres_cosh
        thres_emb = model.cfg.hat.thres_emb
        anneal_steps = model.cfg.hat.anneal_steps
        freeze_task_embedding = model.cfg.freeze_task_embedding

        if anneal_steps == -1:
            anneal_steps = self.num_batches
        
        for name, param in model.named_parameters():
            m = _BLOCK_NAME_RE.match(name)
            if m is None:
                continue
            block_name, layer_type, layer_index, para_type = m.groups()
            block_name = block_name + '.ffn_hat'

            # mask the gradient
            if block_name in model.previous_mask.keys() and param.grad is not None: 
                if para_type == "weight":
                    if layer_index == "2":
                        param.grad *= model.previous_mask[block_name].view(1, -1).expand_as(param.grad)
                    elif layer_index == "1":
                        param.grad *= model.previous_mask[block_name].T.view(-1, 1).expand_as(param.grad)
                    else:
                        raise ValueError("unknown layer:{}.{}.{}".format(block_name,layer_type, layer_index))
                elif para_type == "bias":
                    if layer_index == '1':
                        param.grad *= model.previous_mask[block_name].view(-1)
                else:
                    raise ValueError("unknown parameter:{}.{}.{}".format(block_name,layer_type,para_type))

        for name, param in model.named_parameters():
            if "task_embedding" in name and param.grad is not None:

                # if we are freezing the task embedding, we just zero out the gradient
                if freeze_task_embedding:
                    param.grad.data *= 0.0
                    continue
                                
                # scale the gradient for task embedding
                num = torch.cosh(torch.clamp(temp * param.data, -thres_cosh,
                                             thres_cosh)) + 1
                den = torch.cosh(param.data) + 1
                param.grad.data *= temp_max / temp * num / den

        # anneal the temperature

        current_update = update_num % anneal_steps
        current_epoch = update_num // anneal_steps + 1
        
        temp = temp_min + (temp_max - temp_min) * current_update / (anneal_steps - 1)

        model.cfg.hat.temperature = min(temp, model.cfg.hat.temperature_max)
        
        optimizer.step()

        # clamp the task embedding
        for name, param in model.named_parameters():
            if "task_embedding" in name:
                param.data = torch.clamp(param.data, -thres_emb, thres_emb)
                    
                    

