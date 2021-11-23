#!/bin/bash

python -m exp.run_sr_exp \
  --start_seed 0 \
  --stop_seed 4 \
  --device 0 \
  --exp_name cwn-sr-base \
  --model mp_agnostic \
  --drop_rate 0.0 \
  --graph_norm id \
  --nonlinearity elu \
  --readout sum \
  --final_readout sum \
  --lr_scheduler None \
  --emb_dim 256 \
  --batch_size 8 \
  --num_workers 16 \
  --task_type isomorphism \
  --eval_metric isomorphism \
  --max_ring_size $1 \
  --init_method sum \
  --preproc_jobs 64 \
  --untrained
