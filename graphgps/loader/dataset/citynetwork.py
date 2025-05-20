import os
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset

class citynetwork(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'node_features.pt', '10-chunk_16-hop_node_labels.pt', 'edge_indices.pt',
            'train_mask.pt', 'valid_mask.pt', 'test_mask.pt'
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass 

    def process(self):
        x = torch.load(os.path.join(self.raw_dir, 'node_features.pt'))
        y = torch.load(os.path.join(self.raw_dir, '10-chunk_16-hop_node_labels.pt'))
        edge_index = torch.load(os.path.join(self.raw_dir, 'edge_indices.pt'))

        train_mask = torch.load(os.path.join(self.raw_dir, 'train_mask.pt'))
        val_mask = torch.load(os.path.join(self.raw_dir, 'valid_mask.pt'))
        test_mask = torch.load(os.path.join(self.raw_dir, 'test_mask.pt'))

        data = Data(x=x, y=y, edge_index=edge_index,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data]) 
        torch.save((data, slices), self.processed_paths[0])

class EgoGraphDataset(InMemoryDataset):
    def __init__(self, graphs, split_idxs=None, transform=None, pre_transform=None):
        self.graphs = graphs
        super().__init__('.', transform, pre_transform)
        self.data, self.slices = self.collate(graphs)
        self.split_idxs = split_idxs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]