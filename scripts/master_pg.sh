#!/bin/bash
for dataset in banana waveform breast_cancer image
do
	echo "running on "$dataset
	for num in 1 2 5 10 15 20
	do	
		for reg_exp in 1.0 2.0 5.0 10.0
		do
			for reg_gat in 1.0 2.0 5.0 10.0
			do
				echo $num" expert(s) in use with regularization values ("$reg_exp"," $reg_gat")"
				for rep in {1..50}
				do 
					python mbsvme_pg.py -f True -d $dataset -k $num -re $reg_exp -rg $reg_gat --max_iter 60
				done
			done
		done
	done
done
