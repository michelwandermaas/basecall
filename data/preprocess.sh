#!/bin/bash

if [ $# -eq 1 ] && [ $1 == '--help' ]; then
	echo 'USAGE: ./script input_file  output_folder'
	echo "The default NUMBER_PS_NODES is 1."
	exit
elif [ $# -eq 2 ]; then
	INPUT_FILE=$1
	OUTPUT_FOLDER=$2
else
        echo 'Wrong number of arguments, use --help to see the usage.'
        exit
fi


awk '{print $1;}' $INPUT_FILE | head -n 3 > file_names
mkdir train
mkdir train/reads
cat file_names | while read line
do
	poretools events $line > train/$line
	awk '{print $3,$5,$6;}' train/$line > train/$line.clean
done
awk '{print $2;}' odd.train | head -n 3 > ref_seq
for i in {1..3}
do
	eval FILE=`sed -n -e ${i}p file_names`
	sed -n -e ${i}p ref_seq > ref${i}_seq
	CURRENT_DIR=`pwd`
	echo $CURRENT_DIR/$FILE > file${i}_name
	paste train/$FILE.clean ref${i}_seq file${i}_name > train${i}.input
done
rm file_names
rm -rf train
mkdir $OUTPUT_FOLDER
mv *.input $OUTPUT_FOLDER
