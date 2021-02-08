import sys
import copy
import time
import numpy as np
from exp.parser import get_parser
from exp.run_exp import main

# python3 -m exp.run_tu_exp --dataset IMDBBINARY --model sin --drop_rate 0.0 --lr 0.0001 --max_dim 2 --emb_dim 32 --dump_curves --epochs 30 --num_layers 1 --lr_scheduler StepLR --lr_scheduler_decay_steps 5

__num_folds__ = 10

if __name__ == "__main__":
    
    # standard args
    parser = get_parser()
    passed_args = sys.argv[1:]
    assert 'fold' not in passed_args
    passed_args += ['--exp_name', str(time.time())]
    parsed_args = parser.parse_args(passed_args)

    # run each experiment separately and gather results
    results = list()
    for i in range(__num_folds__):
        current_args = copy.copy(passed_args) + ['--fold', str(i)]
        parsed_args = parser.parse_args(current_args)
        curves = main(parsed_args)
        results.append(curves)
        
    # aggregate results
    train_curves = np.asarray([curves['train'] for curves in results])
    val_curves = np.asarray([curves['val'] for curves in results])
    avg_train_curve = train_curves.mean(axis=0)
    avg_val_curve = val_curves.mean(axis=0)
    best_index = np.argmax(avg_val_curve)
    mean_perf = avg_val_curve[best_index]
    std_perf = val_curves.std(axis=0)[best_index]
    
    msg = (
        'Dataset:        {0}\n'
        'Accuracy:       {1} ± {2}\n'
        'Best epoch:     {3}\n'
        '-------------------------------\n').format(parsed_args.dataset,
                                                    mean_perf, std_perf, best_index)
    print(msg)
    
        
