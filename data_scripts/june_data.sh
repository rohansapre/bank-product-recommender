#!/bin/bash

file='../input/train_ver2.csv'
outFile='../input/june_filter_data.csv'
rm $outFile
head -1 $file >> $outFile
awk -F ',' '$1 == "2015-05-28" || $1 == "2015-06-28"' $file >> $outFile
python ../findNewProdCustomers.py
python ../predict.py
