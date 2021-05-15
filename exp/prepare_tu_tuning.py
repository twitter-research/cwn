import sys
import yaml
from data.data_loading import load_dataset

if __name__ == "__main__":
    
    # standard args
    passed_args = sys.argv[1:]
    conf_path = passed_args[0]
    
    # parse grid from yaml
    with open(conf_path, 'r') as handle:
        conf = yaml.safe_load(handle)
    dataset = conf['dataset']
    max_dims = conf['max_dim']
    max_ring_sizes = conf['max_ring_size']
    init_methods = conf['init_method']
    
    # build datasets is not present
    for max_dim in max_dims:
        for max_ring_size in max_ring_sizes:
            for init in init_methods:
                _ = load_dataset(dataset, max_dim=max_dim, init_method=init, max_ring_size=max_ring_size)
