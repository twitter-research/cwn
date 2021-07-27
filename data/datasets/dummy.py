import torch

from data.datasets import InMemoryComplexDataset
from data.dummy_complexes import get_testing_complex_list, get_mol_testing_complex_list


class DummyDataset(InMemoryComplexDataset):
    """A dummy dataset using a list of hand-crafted cell complexes with many edge cases."""

    def __init__(self, root):
        self.name = 'DUMMY'
        super(DummyDataset, self).__init__(root, max_dim=3, num_classes=2,
            init_method=None, include_down_adj=True, cellular=False)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_ids = list(range(self.len()))
        self.val_ids = list(range(self.len()))
        self.test_ids = list(range(self.len()))
            
    @property
    def processed_file_names(self):
        name = self.name
        return [f'{name}_complex_list.pt']
    
    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return []
    
    def download(self):
        return
    
    @staticmethod
    def factory():
        complexes = get_testing_complex_list()
        for c, complex in enumerate(complexes):
            complex.y = torch.LongTensor([c % 2])
        return complexes
        
    def process(self):
        print("Instantiating complexes...")
        complexes = self.factory()
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])


class DummyMolecularDataset(InMemoryComplexDataset):
    """A dummy dataset using a list of hand-crafted molecular cell complexes with many edge cases."""

    def __init__(self, root, remove_2feats=False):
        self.name = 'DUMMYM'
        self.remove_2feats = remove_2feats
        super(DummyMolecularDataset, self).__init__(root, max_dim=2, num_classes=2,
            init_method=None, include_down_adj=True, cellular=True)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_ids = list(range(self.len()))
        self.val_ids = list(range(self.len()))
        self.test_ids = list(range(self.len()))
            
    @property
    def processed_file_names(self):
        name = self.name
        remove_2feats = self.remove_2feats
        fn = f'{name}_complex_list'
        if remove_2feats:
            fn += '_removed_2feats'
        fn += '.pt'
        return [fn]
    
    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return []
    
    def download(self):
        return
    
    @staticmethod
    def factory(remove_2feats=False):
        complexes = get_mol_testing_complex_list()
        for c, complex in enumerate(complexes):
            if remove_2feats:
                if 2 in complex.cochains:
                    complex.cochains[2].x = None
            complex.y = torch.LongTensor([c % 2])
        return complexes
        
    def process(self):
        print("Instantiating complexes...")
        complexes = self.factory(self.remove_2feats)
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
