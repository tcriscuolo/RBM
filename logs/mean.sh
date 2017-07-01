#!/bin/bash

folder=$1 
what="changes" 
out="${folder}/$what.txt"

rm -f $out

for i in `seq 1 20`
do
	echo "$what ${folder}/execution_${i}.log >> $out"
	grep $what ${folder}/execution_${i}.log | awk '{print $NF}' >> $out  
done
