#!/bin/bash
for dataset in landmine
do
	echo "running on "$dataset
	for num in 1 2 5 10
	do
		for reg_exp in 10.0 1.0 5.0 20.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..18}
			do
				for rep in {1..5}
				do
					python mbsvme_gen.py -f True -d $dataset -k $num -r $reg_exp --max_iters 50 -p gauss -t $taskid
				done
			done
		done
	done
done
