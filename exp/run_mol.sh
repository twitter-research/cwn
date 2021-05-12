#!/bin/bash

# Reference example for running ZINC
python -m exp.run_exp --dataset ZINC --train_eval_period 10 --epochs 10 --batch_size 32 \
  --drop_rate 0.5 --drop_position lin2 --emb_dim 32 --max_dim 2 --final_readout sum \
  --init_method mean --jump_mode cat --lr 0.003 --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.5 --lr_scheduler_decay_steps 50 --model zinc_sparse_sin \
  --nonlinearity relu --num_layers 2 --readout mean --max_ring_size=5 --task_type=regression \
  --eval_metric=mae --minimize --lr_scheduler='ReduceLROnPlateau'
