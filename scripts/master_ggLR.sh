#!/bin/bash
for dataset in breast_cancer
do
	echo "running on "$dataset
	for num in 5 10 20
	do
		for reg_exp in 10.0 1.0 5.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for rep in {0..9}
			do
				python3 ggLR.py -f True -d $dataset -k $num -r $reg_exp --max_iters 50 -p gauss
			done
		done
	done
done