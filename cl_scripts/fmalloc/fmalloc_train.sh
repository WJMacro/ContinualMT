DEVICE=$1
MAX_TEMP=$2
MASK_LAMBDA=$3
SPARSITY=$4
SEQ_ID=$5
# min temperature should be 1 / max temperature
MIN_TEMP=$(echo "scale=4; 1.0 / $MAX_TEMP" | bc)

export CUDA_VISIBLE_DEVICES=$DEVICE

CKPT_DIR=outputs/transformer-fmalloc-${MAX_TEMP}-${MASK_LAMBDA}-${SPARSITY}-seq-${SEQ_ID}

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR

PT_MODEL_DIR=pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt
IMPORTANCE_DIR=outputs/transformer-ffn-importance/importance.pt

TASKID=1
# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_${SEQ_ID}.txt)
# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do

    # train on current dataset
    echo "Training on $DATASET"
    python train.py data-bin/$DATASET \
        --task fmalloc_translation \
        --user-dir approaches \
        --save-dir $CKPT_DIR \
        --arch fmalloc@transformer_wmt19_de_en \
        --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy_with_capacity \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --max-update 300000 \
        --warmup-updates 4000 \
        --patience 5 \
        --validate-interval 999 --save-interval 999 \
        --validate-interval-updates 1000 --keep-interval-updates 1 \
        --save-interval-updates 1000 \
        --no-epoch-checkpoints \
        --no-last-checkpoints \
        --skip-invalid-size-inputs-valid-test \
        --tensorboard-logdir $CKPT_DIR/tensorboard \
        --restore-file $CKPT_DIR/checkpoint_best.pt \
        --reset-optimizer --reset-dataloader --reset-meters \
        --reset-lr-scheduler \
        --pretrained-transformer-path $PT_MODEL_DIR \
        --general-domain-mask-path $IMPORTANCE_DIR \
        --enable-hat \
        --hat-task-num 6 --hat-task-id $TASKID \
        --hat-temperature $MIN_TEMP --hat-temperature-max $MAX_TEMP \
        --hat-temperature-min $MIN_TEMP \
        --hat-anneal-steps 1000 \
        --sparsity $SPARSITY 
    
    
    TASKID=$((TASKID+1))

    # test on previous datasets
    PREV_TASKID=0       
    while [ $PREV_TASKID -lt $TASKID ]
    do
        TEST_DATASET=$(echo general_test $TASK_SEQ | cut -d' ' -f$((PREV_TASKID + 1)))
        # test on TEST_DATASET
        echo "Testing on $TEST_DATASET with task id $PREV_TASKID"
        python fairseq_cli/generate.py data-bin/$TEST_DATASET \
            --path $CKPT_DIR/checkpoint_best.pt \
            --task fmalloc_translation \
            --arch fmalloc@transformer_wmt19_de_en \
            --user-dir approaches \
            --gen-subset test \
            --quiet \
            --beam 5 --remove-bpe \
            --max-len-b 10 --max-len-a 1.2 \
            --model-overrides "{'hat_task_id': $PREV_TASKID, 'hat_temperature':$MAX_TEMP}"

        PREV_TASKID=$((PREV_TASKID+1))
    done

done
