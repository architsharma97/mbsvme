#!/bin/bash
for dataset in parkinsons
do
	echo "running on "$dataset
	for num in 10 5 1 20 5
	do
		for reg_exp in 1.0 2.0 10.0 5.0
		do
			for reg_gat in 2.0 10.0 1.0 5.0
			do
				echo $num" expert(s) in use with regularization values ("$reg_exp"," $reg_gat")"
				for rep in {1..2}
				do
					python mbsvme_pg.py -f True -d $dataset -k $num -re $reg_exp -rg $reg_gat --max_iter 70
				done
			done
		done
	done
done
