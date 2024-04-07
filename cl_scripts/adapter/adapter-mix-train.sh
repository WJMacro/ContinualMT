DEVICE=$1

export CUDA_VISIBLE_DEVICES=$DEVICE

python fairseq_cli/train.py data-bin/mixed-domain \
    --ddp-backend=no_c10d \
    --task translation \
    --user-dir approaches \
    --save-dir checkpoints/adapter-mix \
    --arch adapter@transformer_wmt_de_en \
    --share-decoder-input-output-embed \
    --decoder-normalize-before \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --warmup-updates 4000 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 100 \
    --no-epoch-checkpoints \
    --no-save-optimizer-state \
    --patience 10 \
    --skip-invalid-size-inputs-valid-test \
    --pretrained-transformer-path pretrain-models/wmt17.de-en/checkpoint_best.pt \
    --adapter-bottelneck-dim 512 \


# enumerate all datasets it koran law medical
for DATASET in it koran law medical subtitles
do
    # test on TEST_DATASET
    echo "Testing on $DATASET"
    python fairseq_cli/generate.py data-bin/$DATASET \
        --path checkpoints/adapter-mix/checkpoint_best.pt \
        --task translation \
        --arch adapter@transformer_wmt_de_en \
        --user-dir approaches \
        --gen-subset test \
        --quiet \
        --beam 5 --remove-bpe \
        --max-len-b 10 --max-len-a 1.2 
done


