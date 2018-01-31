#!/bin/bash
for dataset in ijcnn
do
	echo "running on "$dataset
	for num in 1 5 10 20
	do	
		for reg_exp in 1.0 2.0 10.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for rep in {1..1}
			do 
				python mbsvme_gen.py -f True -d $dataset -k $num -r $reg_exp --max_iters 40
			done
		done
	done
done