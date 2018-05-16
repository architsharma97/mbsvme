#!/bin/bash
for dataset in adult
do
	echo "running on "$dataset
	for num in 6 3
	do
		for reg_exp in 1.0 5.0 10.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..0}
			do
				for rep in {1..2}
				do
					python hme.py -f True -d $dataset -l $num -r $reg_exp --max_iters 60 -p gauss -t $taskid
				done
			done
		done
	done
done
