#!/bin/bash

# Reference example for running ZINC
python -m exp.run_exp --dataset ZINC --train_eval_period 10 --epochs 300 --batch_size 64   \
  --drop_rate 0.5 --drop_position lin2 --emb_dim 128 --max_dim 2 --final_readout sum \
  --init_method sum --jump_mode cat --lr 0.003 --model zinc_sparse_sin --nonlinearity relu \
  --num_layers 3 --readout mean --max_ring_size=6 --task_type=regression --eval_metric=mae \
  --minimize --lr_scheduler='ReduceLROnPlateau'
