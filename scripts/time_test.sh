#!/bin/bash
for h in 2 5 10 15 20 
do
	for rep in {0..20}
	do
		{ time python3 mbsvme_gg.py -k $h -d image -m 100 ; } 2>> "../results/times/va-gg"$h"image.txt"
	done
done