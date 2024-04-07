#!/bin/bash 

# the pre-trained general-domain checkpoint
ckt=checkpoints/pte_general/checkpoint_best.pt

rm -rf checkpoints/pte_pruned_general
mkdir -p checkpoints/pte_pruned_general

# path to save the pruned checkpoint
save_ckt=checkpoints/pte_pruned_general/checkpoint.pt

save_mask=checkpoints/pte_pruned_general/mask0.pt

python magnitude.py --pre-ckt-path $ckt --save-ckt-path $save_ckt \
            --save-mask-path $save_mask --prune-ratio 0.20 \
            # --only-mask

TASKID=1
# prune ratio
for ratio in 0.16 0.12 0.08 0.04 0
do
    # path to save the mask matrix 
    save_mask=checkpoints/pte_pruned_general/mask${TASKID}.pt

    python magnitude.py --pre-ckt-path $ckt --save-ckt-path $save_ckt \
                --save-mask-path $save_mask --prune-ratio $ratio \
                --only-mask

    TASKID=$((TASKID+1))
    
done