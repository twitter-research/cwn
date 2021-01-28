import sys
import copy
import time
from exp.parser import get_parser
from exp.run_exp import main

__num_folds__ = 10

if __name__ == "__main__":
    
    # standard args
    parser = get_parser()
    passed_args = sys.argv[1:]
    assert 'fold' not in passed_args
    passed_args + ['--exp_name', str(time.time())]
    
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
    # TODO: check whether we need to pick the max on val or train
    best_index = np.argmax(avg_val_curve)
    mean_perf = avg_val_curve[best_idx]
    std_perf = val_curves.std(axis=0)[best_idx]
    
    msg = (
        'Dataset:        {0}\n'
        'Accuracy:       {1} ± {2}\n'
        'Best epoch:     {3}\n'
        '-------------------------------\n').format(args.dataset, mean_perf, std_perf, best_index)
    print(msg)
    
        
    