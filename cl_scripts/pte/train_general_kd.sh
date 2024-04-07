DEVICE=$1

export CUDA_VISIBLE_DEVICES=$DEVICE

CKPT_DIR=checkpoints/pte_general_kd

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR

MASK_DIR=checkpoints/pte_pruned_general/mask0.pt

python fairseq_cli/train.py data-bin/wmt17_de_en \
    --task pte_translation \
    --user-dir approaches \
    --save-dir $CKPT_DIR \
    --arch transformer_wmt19_de_en \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion rec_loss \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --warmup-updates 4000 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 1 \
    --patience 5 \
    --validate-interval 999 --save-interval 999 \
    --validate-interval-updates 1000 --keep-interval-updates 1 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --skip-invalid-size-inputs-valid-test \
    --enable-knowledge-distillation \
    --finetune-from-model checkpoints/pte_pruned_general/checkpoint.pt \
    --teacher-model-path checkpoints/pte_general/checkpoint_best.pt \
    --tunable-mask-path $MASK_DIR 


TEST_DATASET=general_test

python fairseq_cli/generate.py data-bin/$TEST_DATASET \
    --path checkpoints/pte_general_kd/checkpoint_best.pt \
    --task translation \
    --arch transformer_wmt19_de_en \
    --user-dir approaches \
    --gen-subset test \
    --quiet \
    --beam 5 --remove-bpe \
    --max-len-b 10 --max-len-a 1.2 