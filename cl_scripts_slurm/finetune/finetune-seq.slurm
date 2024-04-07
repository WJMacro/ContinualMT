#!/bin/bash
#SBATCH --job-name finetune_seq
#SBATCH --partition q_ai
#SBATCH --output=./logs/finetune-seq-0.log
#SBATCH --error=./logs/finetune-seq-0.log
#SBATCH --nodelist=g05
#SBATCH --gres=gpu:1

SEQ_ID=0
TASKID=0
CKPT_DIR=checkpoints/fintune-seq-$SEQ_ID

rm -rf $CKPT_DIR
mkdir -p $CKPT_DIR
cp pretrained_models/wmt19.de-en.joined-dict.ensemble/model1.pt $CKPT_DIR/checkpoint_best.pt

# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_${SEQ_ID}.txt)
# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do

    # last task id
    LAST_TASKID=$((TASKID-1))

    # train on current dataset
    echo "Training on $DATASET"
    python fairseq_cli/train.py data-bin/$DATASET \
        --task translation \
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
        --max-update 100000 \
        --no-epoch-checkpoints \
        --no-save-optimizer-state \
        --patience 5 \
        --validate-interval 999 --save-interval 999 \
        --validate-interval-updates 1000 --keep-interval-updates 1 \
        --save-interval-updates 1000 \
        --skip-invalid-size-inputs-valid-test \
        --no-last-checkpoints \
        --finetune-from-model $CKPT_DIR/checkpoint_best.pt 
    
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
            --task translation \
            --arch transformer_wmt19_de_en \
            --gen-subset test \
            --quiet \
            --beam 5 --remove-bpe \
            --max-len-b 10 --max-len-a 1.2 
            
        PREV_TASKID=$((PREV_TASKID+1))
    done

done