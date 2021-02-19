#!/bin/bash

for i in $( seq 0 3 )
do
    python3 -m exp.run_tu_tuning --conf ~/git/scn/exp/tuning_configurations/RDTB_grid.yml --code reddditbinary_20210219 --idx $i &
done
