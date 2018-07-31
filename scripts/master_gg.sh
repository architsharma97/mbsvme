#!/bin/bash
for dataset in wisconsin
do
	echo "running on "$dataset
	for num in 10 5 20
	do
		for reg_exp in 10.0 1.0 5.0 20.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..0}
			do
				for rep in {0..0}
				do
					python mbsvme_gg.py -f True -d $dataset -k $num -r $reg_exp --max_iters 60 -p gauss -t $taskid
				done
			done
		done
	done
done