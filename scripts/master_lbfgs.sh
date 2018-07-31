#!/bin/bash
ds=$1
for dataset in $ds
do
	echo "running on "$dataset
	for num in 10 5 20
	do
		for reg_exp in 1.0 5.0 10.0
		do
			for reg_gat in 0.01 1.0
			do
				echo $num" expert(s) in use with regularization values ("$reg_exp"," $reg_gat")"
				for rep in {0..5}
				do
					python3 mbsvme_gd.py -f True -d $dataset -k $num -re $reg_exp -rg $reg_gat --max_iter 50
				done
			done
		done
	done
done
