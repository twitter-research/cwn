#!/bin/bash

python -m exp.run_mol_exp \
  --stop_seed 4 \
  --epochs 250 \
  --dataset OCEAN \
  --model edge_orient \
  --num_layers 4 \
  --emb_dim 64 \
  --lr 0.001 \
  --batch_size=64 \
  --preproc_jobs 2 \
  --test_orient random \
  --nonlinearity "$1" \
  --drop_rate 0.0 \
  --lr_scheduler_decay_steps 50 \
  --exp_name ocean_mpsn \
  --dump_curves
