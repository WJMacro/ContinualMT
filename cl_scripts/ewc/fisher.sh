DEVICE=$1

export CUDA_VISIBLE_DEVICES=$DEVICE

TASKID=0
lambda=5000
CKPT_DIR=checkpoints/transformer-ewc
TEST_DATASET=medical

python fairseq_cli/compute_fisher.py data-bin/$TEST_DATASET \
    --task translation \
    --user-dir approaches \
    --save-dir $CKPT_DIR \
    --arch transformer_wmt19_de_en \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --max-epoch 1 \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --skip-invalid-size-inputs-valid-test \
    --restore-file $CKPT_DIR/checkpoint_best.pt \