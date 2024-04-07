DEVICE=$1
SEQ_ID=$2

export CUDA_VISIBLE_DEVICES=$DEVICE

CKPT_DIR=checkpoints/transformer-pte-seq-${SEQ_ID}

TASKID=1

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR
cp checkpoints/pte_general_kd/checkpoint_best.pt $CKPT_DIR/checkpoint_best.pt

# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_${SEQ_ID}.txt)
# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do

    # train on current dataset
    echo "Training on $DATASET"
    python fairseq_cli/train.py data-bin/$DATASET \
        --task pte_translation \
        --user-dir approaches \
        --save-dir $CKPT_DIR \
        --arch transformer_wmt19_de_en \
        --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
        --lr 1e-4 --lr-scheduler inverse_sqrt \
        --weight-decay 0.0001 \
        --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --max-epoch 100 \
        --no-epoch-checkpoints \
        --no-save-optimizer-state \
        --patience 5 \
        --validate-interval 999 --save-interval 999 \
        --validate-interval-updates 1000 --keep-interval-updates 1 \
        --save-interval-updates 1000 \
        --skip-invalid-size-inputs-valid-test \
        --no-last-checkpoints \
        --finetune-from-model $CKPT_DIR/checkpoint_best.pt \
        --freeze-mask-path checkpoints/pte_pruned_general/mask$((TASKID-1)).pt \
        --tunable-mask-path checkpoints/pte_pruned_general/mask$TASKID.pt \
    

    # test on TEST_DATASET
    echo "Testing on $DATASET"
    python fairseq_cli/generate.py data-bin/$DATASET \
        --path $CKPT_DIR/checkpoint_best.pt \
        --task translation \
        --arch transformer_wmt19_de_en \
        --user-dir approaches \
        --gen-subset test \
        --quiet \
        --beam 5 --remove-bpe \
        --max-len-b 10 --max-len-a 1.2 

    TASKID=$((TASKID+1))

done


