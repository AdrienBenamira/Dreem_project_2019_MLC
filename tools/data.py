import os
import numpy as np
import h5py
import csv
import torch
from typing import List, Tuple

__all__ = ['data_transformer', 'DreemDataset', 'DreemDatasets']


def split_train_validation(len_dataset: int, percent_train: float) -> Tuple[List[int], List[int]]:
    """
    Splits between train set and validation set
    Args:
        len_dataset: size of the data set
        percent_train: splits according to this percentage.

    Returns: couple of indexes for train set and validation set

    """
    items = np.arange(0, len_dataset, dtype=np.int)
    np.random.shuffle(items)
    split = int(percent_train * len_dataset)
    return items[:split], items[split:]


class DreemDatasets:
    """
    Context Manager for DreepDataset to open and close datasets properly
    """
    def __init__(self, data_path: str, target_path: str = None, keep_datasets: List[str] = None,
                 split_train_val: float = 0.8):
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
            split_train_val: percentage of dataset to keep for the training set
        """
        self.data_path = data_path
        self.target_path = target_path
        self.keep_datasets = keep_datasets
        self.split_train_val = split_train_val

    def __enter__(self):
        # Start by initialising the first one to get the size of the dataset
        self.train = DreemDataset(self.data_path, self.target_path, self.keep_datasets)
        # Get the split
        keys_train, keys_val = split_train_validation(len(self.train), self.split_train_val)
        self.train.set_keys_to_keep(keys_train)
        # Initialize the second one
        self.val = DreemDataset(self.data_path, self.target_path, self.keep_datasets, keys_val)
        return self.train, self.val

    def __exit__(self, *args):
        self.train.close()
        self.val.close()


class DreemDataset:
    targets = None
    length = None
    datasets = {}

    def __init__(self, data_path: str, target_path: str = None, keep_datasets: List[str] = None,
                 keys_to_keep: List[int] = None):
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
            keys_to_keep: keys of the dataset to keep
        """
        self.keys_to_keep = keys_to_keep
        self.keep_datasets = keep_datasets
        self._load_data(data_path)
        if target_path is not None:
            self._load_target(target_path)
        self.length = self.length if keys_to_keep is None else len(keys_to_keep)

    def set_keys_to_keep(self, keys_to_keep: List[int]):
        self.keys_to_keep = keys_to_keep
        self.length = self.length if keys_to_keep is None else len(keys_to_keep)

    def _load_data(self, data_path):
        path = os.path.abspath(os.path.join(os.curdir, data_path))
        self.data = h5py.File(path, 'r')
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

    def close(self):
        self.data.close()

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        item = item if self.keys_to_keep is None else self.keys_to_keep[item]
        data = []
        for dataset in self.datasets.values():
            data.extend(dataset[item])
        return (torch.tensor(data), self.targets[item]) if self.targets is not None else torch.tensor(data)

    def __len__(self):
        return self.length


def data_transformer():
    return []
