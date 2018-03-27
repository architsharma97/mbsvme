#!/bin/bash
for dataset in banana breast_cancer ijcnn waveform image adult
do
	echo "running on "$dataset
	for num in 2 3 4 5 6
	do
		for reg_exp in 10.0 1.0 5.0 20.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..0}
			do
				for rep in {1..5}
				do
					python mbsvme_gen.py -f True -d $dataset -l $num -r $reg_exp --max_iters 50 -p gauss -t $taskid
				done
			done
		done
	done
done