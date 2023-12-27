#!/bin/bash

# enumerate all the domains

mkdir -p ./general_test

for domain in wmt19_de_en wmt20_de_en wmt21_de_en
do 
    for prefix in train.bpe.filtered dev.bpe test.bpe train.bpe
    do
        for lang in de en
        do
            cat ./$domain/$prefix.$lang >> ./general_test/$prefix.$lang
        done
    done
done