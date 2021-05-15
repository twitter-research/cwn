#!/bin/bash
low=0
high=7
gridpath="<path_to_grid>"
expname="<exp_name>"
python3 -m exp.build_dataset
for i in $( seq $low $high )
do
    python3 -m exp.run_tu_tuning --conf $gridpath --code $expname --idx $i &
done
