#!/bin/bash

# wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

# Usage: bash prepare-domadap.sh medical

DATADIR=$1

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_CODE=fairseq/pretrain-models/wmt17.de-en/code

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

find ${DATADIR} -name "*.tok*" | xargs rm -rf
find ${DATADIR} -name "*.bpe*" | xargs rm -rf

filede=${DATADIR}/train.de
fileen=${DATADIR}/train.en

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l de  >> ${DATADIR}/train.tok.de

cat $fileen | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l en  >> ${DATADIR}/train.tok.en

for split in dev test
do
  filede=${DATADIR}/${split}.de
  fileen=${DATADIR}/${split}.en

  cat $filede | \
    perl $TOKENIZER -threads 8 -a -l de  >> ${DATADIR}/${split}.tok.de

  cat $fileen | \
    perl $TOKENIZER -threads 8 -a -l en  >> ${DATADIR}/${split}.tok.en

done

src=de
tgt=en

for L in $src $tgt; do
    for f in train dev test; do
        echo "apply_bpe.py to ${f}.${L}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < ${DATADIR}/$f.tok.$L >${DATADIR}/$f.bpe.$L
    done
done

perl $CLEAN -ratio 1.5 ${DATADIR}/train.bpe de en ${DATADIR}/train.bpe.filtered 1 250