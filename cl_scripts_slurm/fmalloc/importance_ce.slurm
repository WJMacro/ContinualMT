#!/bin/bash
#SBATCH --job-name test_general
#SBATCH --partition q_ai
#SBATCH --output=./logs/prune-general-valid-ce.log
#SBATCH --error=./logs/prune-general-valid-ce.log
#SBATCH --nodelist=g06
#SBATCH --gres=gpu:1

CKPT_DIR=checkpoints/transformer-ffn-importance-valid-ce

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR
PRETRAINED_MODEL_DIR=pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt

DATASET=wmt17_de_en

python fairseq_cli/compute_hat_importance_eval.py data-bin/$DATASET \
    --train-subset valid \
    --task translation \
    --user-dir approaches \
    --save-dir $CKPT_DIR \
    --arch fmalloc@transformer_wmt19_de_en \
    --activation-dropout 0.2 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion cross_entropy \
    --max-tokens 2048 \
    --max-epoch 1 \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --skip-invalid-size-inputs-valid-test \
    --pretrained-transformer-path $PRETRAINED_MODEL_DIR \
    --calculate-ffn-importance 