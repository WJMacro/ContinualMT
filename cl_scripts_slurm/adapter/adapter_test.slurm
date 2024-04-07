#!/bin/bash
#SBATCH --job-name test_adapter
#SBATCH --partition q_ai
#SBATCH --output=./logs/adapter-128-it-test.log
#SBATCH --error=./logs/adapter-128-it-test.log
#SBATCH --nodelist=g32
#SBATCH --gres=gpu:1

DATASET=it

export CUDA_VISIBLE_DEVICES=0

# test on TEST_DATASET
echo "Testing on $DATASET"
python fairseq_cli/generate.py data-bin/$DATASET \
    --path checkpoints/adapter-$DATASET-128/checkpoint_best.pt \
    --task translation \
    --arch adapter@transformer_wmt19_de_en \
    --user-dir approaches \
    --gen-subset test \
    --quiet \
    --beam 5 --remove-bpe \
    --max-len-b 10 --max-len-a 1.2 


