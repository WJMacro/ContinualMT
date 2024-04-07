#!/bin/bash
#SBATCH --job-name train_adapter
#SBATCH --partition q_ai
#SBATCH --output=./logs/adapter-64-koran.log
#SBATCH --error=./logs/adapter-64-koran.log
#SBATCH --nodelist=g30
#SBATCH --gres=gpu:1

DATASET=koran

# train on current dataset
echo "Training on $DATASET"
python fairseq_cli/train.py data-bin/$DATASET \
    --task translation \
    --user-dir approaches \
    --save-dir checkpoints/adapter-$DATASET-64 \
    --arch adapter@transformer_wmt19_de_en \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --warmup-updates 4000 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 100000 \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --patience 5 \
    --validate-interval 999 --save-interval 999 \
    --validate-interval-updates 1000 --keep-interval-updates 1 \
    --save-interval-updates 1000 \
    --skip-invalid-size-inputs-valid-test \
    --pretrained-transformer-path pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt \
    --adapter-bottelneck-dim 64 \
    # --adapter-enable-layer-norm

# test on TEST_DATASET
echo "Testing on $DATASET"
python fairseq_cli/generate.py data-bin/$DATASET \
    --path checkpoints/adapter-$DATASET-64/checkpoint_best.pt \
    --task translation \
    --arch adapter@transformer_wmt19_de_en \
    --user-dir approaches \
    --gen-subset test \
    --quiet \
    --beam 5 --remove-bpe \
    --max-len-b 10 --max-len-a 1.2 


