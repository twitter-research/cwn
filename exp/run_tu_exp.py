import sys
import copy
from exp.parser import get_parser
from exp.run_exp import main

__num_folds__ = 10

if __name__ == "__main__":
    
    # standard args
    parser = get_parser()
    passed_args = sys.argv[1:]
    assert 'fold' not in passed_args
    
    # run each experiment separately and gather results
    results = list()
    for i in range(__num_folds__):
        current_args = copy.copy(passed_args) + ['--fold', str(i)]
        parsed_args = parser.parse_args(current_args)
        curves = main(parsed_args)
        results.append(curves)
        
    # aggregate results
    
        
    