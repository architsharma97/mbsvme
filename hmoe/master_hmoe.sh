#!/bin/bash
ds=$1
for dataset in $ds
do
	echo "running on "$dataset
	for num in 4 5 6
	do
		for reg_exp in 10.0 1.0 5.0 20.0
		do
			echo $num" expert(s) in use with regularization at "$reg_exp
			for taskid in {0..0}
			do
				for rep in {1..10}
				do
					echo "trial number "$rep
					python hmoe.py -f True -d $dataset -l $num -r $reg_exp --max_iters 60 -p gauss -t $taskid
				done
				echo "------------------------"
                                echo "------------------------" >> "../results/"$ds"_hmoe.txt"
			done
		done
                echo "============================="
		echo "=============================" >> "../results/"$ds"_hmoe.txt"
	done
done
