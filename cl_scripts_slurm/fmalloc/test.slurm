#!/bin/bash
#SBATCH --job-name train_ours_hard
#SBATCH --partition q_ai
#SBATCH --output=./results/fmalloc.out
#SBATCH --error=./results/fmalloc.out
#SBATCH --nodelist=g10
#SBATCH --gres=gpu:1

SEQ_ID=0
MAX_TEMP=400
SPARSITY=0.20
# min temperature should be 1 / max temperature
MIN_TEMP=0.0025


CKPT_DIR=../ARR-23-12/checkpoints/transformer-fmalloc-400-0.20-seq-0

TASKID=0
# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_${SEQ_ID}.txt)

# add general_domain to task sequence
TASK_SEQ="general_test $TASK_SEQ"

mkdir -p ./results/ref
mkdir -p ./results/src
mkdir -p ./results/seq_${SEQ_ID}
mkdir -p ./results/seq_${SEQ_ID}/fmalloc

# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do    
    TASKID=0
    # test on DATASET
    echo "Testing on $DATASET with task id $TASKID"
    python fairseq_cli/generate.py data-bin/$DATASET \
        --path $CKPT_DIR/checkpoint_best.pt \
        --task fmalloc_translation \
        --arch fmalloc@transformer_wmt19_de_en \
        --user-dir approaches \
        --gen-subset test \
        --beam 5 --remove-bpe \
        --max-len-b 10 --max-len-a 1.2 \
        --model-overrides "{'hat_task_id': $TASKID, 'hat_temperature':$MAX_TEMP}" \
        > ./results/seq_${SEQ_ID}/fmalloc/${DATASET}.tmp

    grep ^H ./results/seq_${SEQ_ID}/fmalloc/${DATASET}.tmp | cut -d- -f2- | sort -n | cut -f3- > ./results/seq_${SEQ_ID}/fmalloc/${DATASET}.out

    # detokenize
    perl ../ContinualMT/examples/translation/mosesdecoder/scripts/tokenizer/detokenizer.perl -threads 8 -l en < ./results/seq_${SEQ_ID}/fmalloc/${DATASET}.out > ./results/seq_${SEQ_ID}/fmalloc/${DATASET}.out.detok
    
    TASKID=$((TASKID+1))

done