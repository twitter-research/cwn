#!/bin/bash

python -m exp.run_sr_exp \
  --start_seed 0 \
  --stop_seed 1 \
  --device 0 \
  --exp_name cwn-sr-id-ln-mean-sum-0.0001 \
  --model sparse_cin \
  --use_coboundaries True \
  --drop_rate 0.0 \
  --graph_norm ln \
  --nonlinearity id \
  --readout mean \
  --final_readout sum \
  --lr_scheduler None \
  --num_layers 5 \
  --emb_dim 16 \
  --batch_size 8 \
  --num_workers 16 \
  --task_type isomorphism \
  --eval_metric isomorphism \
  --max_ring_size $1 \
  --init_method sum \
  --preproc_jobs 64 \
  --untrained
