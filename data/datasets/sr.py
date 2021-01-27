import os
import torch
import pickle

from data.sr_utils import load_sr_dataset
from data.utils import compute_clique_complex_with_gudhi
from data.datasets import InMemoryComplexDataset


class SRDataset(InMemoryComplexDataset):

    def __init__(self, root, name, eval_metric, task_type, max_dim=2, num_classes=2,
                 process_args={}, **kwargs):
        process_args['exp_dim'] = max_dim
        super(SRDataset, self).__init__(root, name, eval_metric, task_type, max_dim=max_dim,
                                        num_classes=num_classes, process_args=process_args,
                                        **kwargs)

    def get_idx_split(self):
        # In this dataset, if not explicit split is provided, we don't distinguish between train, val, test sets.
        train_ids = list(range(self.len())) if self._train_ids is None else self._train_ids
        val_ids = list(range(self.len())) if self._val_ids is None else self._val_ids
        test_ids = list(range(self.len())) if self._test_ids is None else self._test_ids
        idx_split = {
            'train': train_ids,
            'valid': val_ids,
            'test': test_ids}
        return idx_split

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pkl'.format(self.name)]

    def process(self):
        data = load_sr_dataset(os.path.join(self.raw_dir, self.name + '.g6'))
        exp_dim = self.process_args['exp_dim']
        complexes = list()
        max_dim = -1
        for datum in data:
            edge_index, num_nodes = datum
            x = torch.ones(num_nodes, 1, dtype=torch.float32)
            complex = compute_clique_complex_with_gudhi(x, edge_index, num_nodes,
                                                        expansion_dim=exp_dim)
            if complex.dimension > max_dim:
                max_dim = complex.dimension
            for dim in range(complex.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = complex.chains[dim].num_features
                else:
                    assert self._num_features[dim] == complex.chains[dim].num_features
            complexes.append(complex)
        self.max_dim = max_dim
        y = torch.ones(len(complexes), dtype=torch.long)
        for complex in complexes:
            complex.y = y
        path = self.processed_paths[0]
        with open(path, 'wb') as handle:
            pickle.dump(complexes, handle)

    def _set_task(self, task_type, num_classes):
        super(SRDataset, self)._set_task(task_type, num_classes)
        self.num_classes = len(self)
