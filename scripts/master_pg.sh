#!/bin/bash
for dataset in banana breast_cancer diabetis flare_solar german heart image ringnorm splice titanic waveform
do
	echo "running on "$dataset
	for num in 1 2 5 10
	do	
		for reg_exp in 1.0 2.0 10.0
		do
			for reg_gat in 1.0 2.0 10.0
			do
				echo $num" expert(s) in use with regularization values ("$reg_exp"," $reg_gat")"
				for rep in {1..50}
				do 
					python mbsvme_pg.py -f True -d $dataset -k $num -re $reg_exp -rg $reg_gat
				done
			done
		done
	done
done