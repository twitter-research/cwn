import itertools
import os
import copy
import yaml
import argparse
from definitions import ROOT_DIR
from exp.parser import get_parser
from exp.run_tu_exp import exp_main

__max_devices__ = 8

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='CWN tuning.')
    parser.add_argument('--conf', type=str, help='path to yaml configuration')
    parser.add_argument('--code', type=str, help='tuning name')
    parser.add_argument('--idx', type=int, help='selection index')
    t_args = parser.parse_args()
    
    # parse grid from yaml
    with open(t_args.conf, 'r') as handle:
        conf = yaml.safe_load(handle)
    dataset = conf['dataset']
    hyper_list = list()
    hyper_values = list()
    for key in conf:
        if key == 'dataset':
            continue
        hyper_list.append(key)
        hyper_values.append(conf[key])
    grid = itertools.product(*hyper_values)
    exp_queue = list()
    for h, hypers in enumerate(grid):
        if h % __max_devices__ == (t_args.idx % __max_devices__):
            exp_queue.append((h, hypers))
    
    # form args
    base_args = [
        '--device', str(t_args.idx),
        '--task_type', 'classification',
        '--eval_metric', 'accuracy',
        '--dataset', dataset,
        '--result_folder', os.path.join(ROOT_DIR, 'exp', 'results', '{}_tuning_{}'.format(dataset, t_args.code))]
    
    for exp in exp_queue:
        args = copy.copy(base_args)
        addendum = ['--exp_name', str(exp[0])]
        hypers = exp[1]
        for name, value in zip(hyper_list, hypers):
            addendum.append('--{}'.format(name))
            addendum.append('{}'.format(value))
        args += addendum
        exp_main(args)

