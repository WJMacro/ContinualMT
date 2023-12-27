# read task sequence from /task_sequence/seq_${SEQ_ID}.txt
TASK_SEQ=$(cat task_sequence/seq_0.txt)

TASKID=2
lambda=0.999

# enumerate all datasets it koran law medical
for DATASET in $TASK_SEQ
do
    # calculate kd lambda
    # kd_lambda = lambda * (1 - lambda^(task_id - 1)) / (1 - lambda^task_id)
    kd_lambda=$(echo "scale=8; $lambda * (1 - $lambda^($TASKID - 1)) / (1 - $lambda^$TASKID)" | bc -l)

    # train on current dataset
    echo "$kd_lambda"

    TASKID=$((TASKID+1))
done




