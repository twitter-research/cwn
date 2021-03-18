#!/bin/bash

for i in $( seq 0 5 )
do
    # python3 -m exp.run_tu_tuning --conf ~/git/scn/exp/tuning_configurations/RDTB_grid.yml --code redditbinary_20210219 --idx $i &
    # python3 -m exp.run_tu_tuning --conf ~/git/scn/exp/tuning_configurations/RDTB_grid.yml --code redditbinary_20210223 --idx $i &
    python3 -m exp.run_tu_tuning --conf ~/git/scn/exp/tuning_configurations/RDTM_grid.yml --code redditmulti_20210227 --idx $i &
done
