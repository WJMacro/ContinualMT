export CUDA_VISIBLE_DEVICES=$1
DATASET=$2

python train.py data-bin/$DATASET \
    --task translation \
    --save-dir checkpoints/finetune-$DATASET \
    --arch transformer_wmt19_de_en \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --max-epoch 100 \
    --patience 5 \
    --validate-interval 999 --save-interval 999 \
    --validate-interval-updates 1000 --keep-interval-updates 1 \
    --save-interval-updates 1000 \
    --no-epoch-checkpoints \
    --finetune-from-model pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt


echo "Testing on $DATASET"
python fairseq_cli/generate.py data-bin/$DATASET \
    --path checkpoints/finetune-$DATASET/checkpoint_best.pt \
    --task translation \
    --arch transformer_wmt19_de_en \
    --gen-subset test \
    --quiet \
    --beam 5 --remove-bpe \
    --max-len-b 10 --max-len-a 1.2 