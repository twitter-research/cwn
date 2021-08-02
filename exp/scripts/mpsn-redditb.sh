#!/bin/bash

python -m exp.run_tu_exp \
  --device 0 \
  --exp_name mpsn-redditb \
  --dataset REDDITBINARY \
  --train_eval_period 50 \
  --epochs 200 \
  --batch_size 32 \
  --drop_rate 0.0 \
  --drop_position 'final_readout' \
  --emb_dim 64 \
  --max_dim 2 \
  --final_readout sum \
  --init_method mean \
  --jump_mode 'cat' \
  --lr 0.001 \
  --graph_norm id \
  --model sparse_cin \
  --nonlinearity relu \
  --num_layers 4 \
  --readout sum \
  --task_type classification \
  --eval_metric accuracy \
  --lr_scheduler 'StepLR' \
  --lr_scheduler_decay_rate 0.5 \
  --lr_scheduler_decay_steps 50 \
  --use_coboundaries False \
  --dump_curves \
  --preproc_jobs 4