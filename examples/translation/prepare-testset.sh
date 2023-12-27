#!/bin/bash

# wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

# Usage: bash prepare-domadap.sh medical

DATADIR=$1
HOME=examples/translation
if [ -z $HOME ]
then
  echo "HOME var is empty, please set it"
  exit 1
fi
SCRIPTS=$HOME/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
FASTBPE=$HOME/fastBPE
BPECODES=pretrained_models/wmt19.de-en.ffn8192/ende30k.fastbpe.code
VOCAB=pretrained_models/wmt19.de-en.ffn8192/dict.en.txt

src=de
tgt=en

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

find ${DATADIR} -name "*.tok*" | xargs rm -rf
find ${DATADIR} -name "*.bpe*" | xargs rm -rf

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' ${DATADIR}/sgm/newstest2020-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > ${DATADIR}/test.$l
    echo ""
done

for split in test
do
  filede=${DATADIR}/${split}.de
  fileen=${DATADIR}/${split}.en

  cat $filede | \
    perl $TOKENIZER -threads 8 -a -l de  >> ${DATADIR}/${split}.tok.de

  cat $fileen | \
    perl $TOKENIZER -threads 8 -a -l en  >> ${DATADIR}/${split}.tok.en

  $FASTBPE/fast applybpe ${DATADIR}/${split}.bpe.de ${DATADIR}/${split}.tok.de $BPECODES $VOCAB
  $FASTBPE/fast applybpe ${DATADIR}/${split}.bpe.en ${DATADIR}/${split}.tok.en $BPECODES $VOCAB
done