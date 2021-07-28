#!/bin/bash

python -m exp.run_mol_exp \
  --stop_seed 4 \
  --epochs 100 \
  --dataset FLOW \
  --model edge_mpnn \
  --num_layers 4 \
  --emb_dim 64 \
  --lr 0.001 \
  --batch_size=64 \
  --flow_points 1000 \
  --preproc_jobs 32 \
  --test_orient random \
  --nonlinearity relu \
  --drop_rate 0.0 \
  --lr_scheduler_decay_steps 20 \
  --exp_name flow_gnn_inv \
  --dump_curves \
  --fully_orient_invar
