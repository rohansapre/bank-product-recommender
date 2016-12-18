#!/bin/bash

rm train_filtered.csv
cut -d ',' -f 1-2,5-6,8,25-48 train_ver2.csv > train_filtered.csv
