#!/bin/bash

infile='../input/train_ver2.csv'
outFile='../input/test_ver3.csv'
rm $outFile
# declare -a indexes=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
# for i in "${indexes[@]}"
# do
# printf '%s\n' | grep '2016-05-28' $file | cut -f$i -d , | perl -p00e 's/\n,/,/g' | paste -sd ',' >> $outFile
# done
IFS=","
while read a b c d e f e f g h i j k l m n o p q r s t u v w x 
do 
	printf '%s\n' | perl -p00e 's/\n,/,/g' | "$a $b $c $d $e $f $e $f $g $h $i $j $k $l $m $n $o $p $q $r $s $t $u $v $w $x" >> $outFile 
done < $infile