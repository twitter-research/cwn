#!/bin/bash

# Reference example for running ZINC
python -m exp.run_exp --dataset ZINC --train_eval_period 10 --epochs 300 --batch_size 64   \
  --drop_rate 0.5 --drop_position lin2 --emb_dim 128 --max_dim 2 --final_readout sum \
  --init_method sum --jump_mode cat --lr 0.003 --model embed_sparse_cin --nonlinearity relu \
  --num_layers 3 --readout mean --max_ring_size 6 --task_type regression --eval_metric mae \
  --minimize --lr_scheduler 'ReduceLROnPlateau'

# Reference example for running ZINC across 10 seeds without edge features
python -m exp.run_mol_exp --dataset ZINC --seeds 10 --train_eval_period 10 \
  --epochs 300 --batch_size 128   --drop_rate 0.0 --drop_position lin2 --emb_dim 128 \
  --max_dim 2 --final_readout sum --init_method sum --lr 0.001 --model embed_sparse_cin \
  --nonlinearity relu --num_layers 4 --readout sum --max_ring_size 10 --task_type regression \
  --eval_metric mae --minimize --lr_scheduler 'ReduceLROnPlateau' --use_coboundaries True

# Reference run for CSL with the GIN params from the Benchmarking GNNs paper
python -m exp.run_mol_exp --seeds 20 --folds 5 --exp_name csl-cin \
  --dataset CSL --train_eval_period 10 --epochs 300 --batch_size 32  --drop_rate 0.1 \
  --drop_position lin2 --emb_dim 100 --max_dim 2 --final_readout sum --init_method sum --lr 5e-4 \
  --model embed_sparse_cin --nonlinearity relu --num_layers 3 --readout sum --max_ring_size 12 \
  --lr_scheduler 'ReduceLROnPlateau' --use_coboundaries True --lr_scheduler_min 1e-6 \
  --lr_scheduler_patience 5 --early_stop

# Reference example for running MOLHIV
python -m exp.run_exp --dataset MOLHIV --train_eval_period 10 --epochs 200 --batch_size 32 \
  --drop_rate 0.5 --drop_position lin2 --emb_dim 300 --max_dim 2 --final_readout sum \
  --init_method sum --lr 0.001 --model ogb_embed_sparse_cin --nonlinearity relu \
  --num_layers 5 --readout mean --max_ring_size 6 --task_type bin_classification \
  --eval_metric ogbg-molhiv --use_coboundaries True --lr_scheduler None --use_edge_features
