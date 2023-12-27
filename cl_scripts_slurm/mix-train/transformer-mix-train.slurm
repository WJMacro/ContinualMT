#!/bin/bash
#SBATCH --job-name train_mix
#SBATCH --partition q_ai
#SBATCH --output=./logs/finetune-mix.log
#SBATCH --error=./logs/finetune-mix.log
#SBATCH --nodelist=g32
#SBATCH --gres=gpu:2

python fairseq_cli/train.py data-bin/mixed-domain \
    --task translation \
    --user-dir approaches \
    --save-dir outputs/transformer-mix \
    --arch transformer_wmt19_de_en \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt \
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
    --validate-interval-updates 4000 --keep-interval-updates 1 \
    --save-interval-updates 4000 \
    --skip-invalid-size-inputs-valid-test \
    --finetune-from-model pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt \


# enumerate all datasets it koran law medical
for DATASET in general_test wmt17_de_en it koran law medical subtitles
do
    # test on TEST_DATASET
    echo "Testing on $DATASET"
    python fairseq_cli/generate.py data-bin/$DATASET \
        --path outputs/transformer-mix/checkpoint_best.pt \
        --task translation \
        --arch hat@transformer_wmt_de_en \
        --user-dir approaches \
        --gen-subset test \
        --quiet \
        --beam 5 --remove-bpe \
        --max-len-b 10 --max-len-a 1.2 
done


