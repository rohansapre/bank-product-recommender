#!/bin/bash

file='../input/train_ver2.csv'
outFile='../input/train_uniq_cols.csv'
rm $outFile
declare -a indexes=(3 4 5 13 14 15 16 17 18)
for i in "${indexes[@]}"
do
printf '%s\n' | sed 1d $file | cut -f$i -d , | sort -u | perl -p00e 's/\n,/,/g' | paste -sd ',' >> $outFile
done
python ../predict.py
