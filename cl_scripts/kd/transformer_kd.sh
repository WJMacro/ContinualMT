DEVICE=$1
SEQ_ID=$2

export CUDA_VISIBLE_DEVICES=$DEVICE

CKPT_DIR=checkpoints/transformer-kd-0.1-test

kd_lambda=0.1
TASKID=0

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR
cp pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt $CKPT_DIR/checkpoint_best.pt

# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_${SEQ_ID}.txt)
# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do
    # train on current dataset
    echo "Training on $DATASET"
    python fairseq_cli/train.py data-bin/$DATASET \
        --task kd_translation \
        --user-dir approaches \
        --save-dir $CKPT_DIR \
        --arch transformer_wmt19_de_en \
        --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
        --lr 5e-5 --lr-scheduler inverse_sqrt \
        --weight-decay 0.0001 \
        --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy_with_dynamic_kd \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --max-epoch 100 \
        --no-epoch-checkpoints \
        --no-save-optimizer-state \
        --patience 10 \
        --validate-interval 999 --save-interval 999 \
        --validate-interval-updates 1000 --keep-interval-updates 1 \
        --save-interval-updates 1000 \
        --skip-invalid-size-inputs-valid-test \
        --no-last-checkpoints \
        --finetune-from-model $CKPT_DIR/checkpoint_best.pt \
        --teacher-model-path $CKPT_DIR/checkpoint_best.pt \
        --kd-lambda $kd_lambda 
    
    TASKID=$((TASKID+1))
    LAST_TASKID=$((TASKID-1))
    # test on previous datasets
    PREV_TASKID=0
    while [ $PREV_TASKID -lt $((TASKID+1)) ]
    do
        TEST_DATASET=$(echo general_test $TASK_SEQ | cut -d' ' -f$((PREV_TASKID + 1)))
        # test on TEST_DATASET
        echo "Testing on $TEST_DATASET"
        python fairseq_cli/generate.py data-bin/$TEST_DATASET \
            --path $CKPT_DIR/checkpoint_best.pt \
            --task kd_translation \
            --arch transformer_wmt19_de_en \
            --user-dir approaches \
            --gen-subset test \
            --quiet \
            --beam 5 --remove-bpe \
            --max-len-b 10 --max-len-a 1.2 

        PREV_TASKID=$((PREV_TASKID+1))

    done

done


