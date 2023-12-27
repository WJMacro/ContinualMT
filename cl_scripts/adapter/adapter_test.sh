DEVICE=$1

export CUDA_VISIBLE_DEVICES=$DEVICE
TMP=tmp

rm -rf $TMP
mkdir -p $TMP
mkdir -p out/adapter-TP-1
# test on all datasetsS
# enumerate all datasets it koran law medical

for DATASET in it koran law medical subtitles
do
    # test on current dataset
    for MODEL in it koran law medical subtitles
    do
        python fairseq_cli/generate.py data-bin/$DATASET \
            --path outputs/adapter-$MODEL-new/checkpoint_best.pt \
            --task translation \
            --user-dir approaches \
            --gen-subset test \
            --beam 1 --remove-bpe \
            --max-len-b 10 --max-len-a 1.2 \
            --no-progress-bar \
            >> $TMP/$DATASET.out.raw
    done

    grep ^H $TMP/$DATASET.out.raw | cut -d- -f2- | sort -n | cut -f2- > $TMP/$DATASET.out.tmp

    python cl_scripts/adapter/select.py --input $TMP/$DATASET.out.tmp --output out/adapter-TP-1/$DATASET.out

    
    grep ^T $TMP/$DATASET.out.raw | cut -d- -f2- | sort -n | cut -f2- > $TMP/$DATASET.ref.tmp
    # drop duplicates
    uniq $TMP/$DATASET.ref.tmp > out/adapter-TP-1/$DATASET.ref
    
done