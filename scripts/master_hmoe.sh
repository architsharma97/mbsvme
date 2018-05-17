#!/bin/bash
for dataset in wisconsin
do
	echo "running on "$dataset
	for num in 3 4 5 6
	do
		for reg_exp in 20.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..0}
			do
				for rep in {1..10}
				do
					python hme.py -f True -d $dataset -l $num -r $reg_exp --max_iters 60 -p gauss -t $taskid
				done
			done
		done
	done
done
