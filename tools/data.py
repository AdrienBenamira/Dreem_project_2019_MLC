import os
import h5py
import csv
import torch

__all__ = ['data_transformer', 'DreemDataset']


class DreemDataset:
    targets = None
    length = None
    datasets = {}

    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path
        self._load_data()
        self._load_target()

    def _load_data(self):
        path = os.path.abspath(os.path.join(os.curdir, self.data_path))
        self.data = h5py.File(path, 'r')
        self.accelerometer_x = self.data['accelerometer_x']
        for item in self.data:
            if self.length is None:
                self.length = self.data[item].shape[0]
            self.datasets[item] = self.data[item]

    def _load_target(self):
        path = os.path.abspath(os.path.join(os.curdir, self.target_path))
        with open(path, 'r') as f:
            reader = csv.reader(f)
            self.targets = {int(i): int(j) for i, j in list(reader)[1:]}

    def __del__(self):
        self.data.close()

    def __getitem__(self, item):
        data = []
        for dataset in self.datasets.values():
            data.extend(dataset[item])
        return torch.tensor(data), self.targets[item]

    def __len__(self):
        return self.length


def data_transformer():
    return []
