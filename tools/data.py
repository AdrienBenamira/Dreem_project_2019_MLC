import os
import h5py
import csv
import torch

__all__ = ['data_transformer', 'DreemDataset']


class DreemDataset:
    targets = None
    length = None
    datasets = {}

    def __init__(self, data_path, target_path=None, keep_datasets=None):
        """
        Args:
            data_path: path to data
            target_path: path to labels, if None, no labels
            keep_datasets: default None, if list, only keep the data sets in the list.
                Available datasets:
                * eeg_1 - EEG in frontal position sampled at 50 Hz -> 1500 values
                * eeg_2 - EEG in frontal position sampled at 50 Hz -> 1500 values
                * eeg_3 - EEG in frontal position sampled at 50 Hz -> 1500 values
                * eeg_4 - EEG in frontal-occipital position sampled at 50 Hz -> 1500 values
                * eeg_5 - EEG in frontal-occipital position sampled at 50 Hz -> 1500 values
                * eeg_6 - EEG in frontal-occipital position sampled at 50 Hz -> 1500 values
                * eeg_7 - EEG in frontal-occipital position sampled at 50 Hz -> 1500 values
                * accelerometer_x - Accelerometer along x axis sampled at 10 Hz -> 300 values
                * accelerometer_y - Accelerometer along y axis sampled at 10 Hz -> 300 values
                * accelerometer_z - Accelerometer along z axis sampled at 10 Hz -> 300 values
                * pulse_oximeter_infrared - Pulse oximeter infrared channel sampled at 10 Hz -> 300 values
        """
        self.keep_datasets = keep_datasets
        self._load_data(data_path)
        if target_path is not None:
            self._load_target(target_path)

    def _load_data(self, data_path):
        path = os.path.abspath(os.path.join(os.curdir, data_path))
        self.data = h5py.File(path, 'r')
        self.accelerometer_x = self.data['accelerometer_x']
        for item in self.data:
            if self.keep_datasets is not None and item in self.keep_datasets:
                if self.length is None:
                    self.length = self.data[item].shape[0]
                self.datasets[item] = self.data[item]

    def _load_target(self, target_path):
        path = os.path.abspath(os.path.join(os.curdir, target_path))
        with open(path, 'r') as f:
            reader = csv.reader(f)
            self.targets = {int(i): int(j) for i, j in list(reader)[1:]}

    def __del__(self):
        self.data.close()

    def __getitem__(self, item):
        data = []
        for dataset in self.datasets.values():
            data.extend(dataset[item])
        return (torch.tensor(data), self.targets[item]) if self.targets is not None else torch.tensor(data)

    def __len__(self):
        return self.length


def data_transformer():
    return []
