# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .fmalloc_transformer import FMALLOCTransformerEncoder, FMALLOCTransformerDecoder, FMALLOCTransformerModel
from .fmalloc_transformer_config import FMALLOCTransformerConfig
from .hat_layer import HATLayer


from fairseq.models import (
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import transformer_legacy

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer")
def base_architecture(args):
    transformer_legacy.base_architecture(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    transformer_legacy.transformer_iwslt_de_en(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    transformer_legacy.base_architecture(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_wmt_de_en")
def transformer_wmt_en_de(args):
    transformer_legacy.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    transformer_legacy.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    transformer_legacy.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    transformer_legacy.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    transformer_legacy.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("fmalloc_transformer", "fmalloc@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    transformer_legacy.transformer_wmt19_de_en(args)


