#!/bin/bash

# enumerate all the domains

mkdir -p ./mixed-domain

for domain in it koran law medical subtitles
do 
    for prefix in train.bpe.filtered dev.bpe test.bpe train.bpe
    do
        for lang in de en
        do
            cat ./$domain/$prefix.$lang >> ./mixed-domain/$prefix.$lang
        done
    done
done