#!/bin/bash
for dataset in banana breast_cancer diabetis flare_solar german heart image ringnorm splice titanic waveform
do
	echo "running on "$dataset
	for num in 1 2 5 10
	do	
		for reg_exp in 1.0 2.0 10.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for rep in {1..50}
			do 
				python mbsvme_gen.py -f True -d $dataset -k $num -r $reg_exp
			done
		done
	done
done