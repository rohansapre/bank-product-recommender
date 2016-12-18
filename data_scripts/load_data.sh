#!/bin/bash

# grep -v "NA" train_ver2.csv >> clean_train.csv

readonly x=12
for i in $(seq 1 18); do
	rm date${i}train.csv
	if [ $i -le 12 ]; then
		head -1 train_ver2.csv >> date${i}train.csv
		grep "2015-$((i/10))$((i%10))-28" clean_train.csv >> date${i}train.csv
		python dataload.py date${i}train "2015-$((i/10))$((i%10))-28"
	else
		head -1 clean_train.csv >> date${i%12}train.csv
		grep "2016-0$((i%12))-28" clean_train.csv >> date${i}train.csv
		python dataload.py date${i}train "2016-0$((i%12))-28"
	fi
done
