#!/bin/bash
ds=$1
for dataset in $ds
do
	echo "running on "$dataset
	for num in 10 5 20
	do
		for reg_exp in 1.0 2.0 10.0 5.0
		do
			for reg_gat in 2.0 10.0 1.0 5.0
			do
				echo $num" expert(s) in use with regularization values ("$reg_exp"," $reg_gat")"
				for id in {0..0}
				do
					for rep in {0..5}
					do
						python mbsvme_pg.py -f True -d $dataset -k $num -re $reg_exp -rg $reg_gat --max_iter 60 -i $id
					done
				done
			done
		done
	done
done
