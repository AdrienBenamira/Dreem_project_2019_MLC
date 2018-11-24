import os
import pickle

import numpy as np
import h5py
import csv
import torch
import pandas as pd
from typing import List, Tuple

__all__ = ['DreemDataset', 'DreemDatasets']


def split_train_validation(len_dataset: int, percent_train: float,
                           seed: float = None, length: int = None,
                           index_labels: dict = None) -> Tuple[List[int], List[int]]:
    """
    Splits between train set and validation set
    Args:
        index_labels:
        len_dataset: size of the data set
        percent_train: splits according to this percentage.
        seed: Seed to use
        length: Size max of the dataset

    Returns: couple of indexes for train set and validation set

    """
    if seed is not None:
        np.random.seed(seed)
    if index_labels is None:
        len_dataset = len_dataset if length is None or length > len_dataset else length
        items = np.arange(0, len_dataset, dtype=np.int)
    else:
        taille_min_sac = min([len(index_labels[k]) for k in range(5)])
        samples = []
        for k in range(5):
            labels = index_labels[k]
            np.random.shuffle(labels)
            samples.append(labels[:taille_min_sac])
        items = np.array([i for j in samples for i in j])
    np.random.shuffle(items)
    split = int(percent_train * len(items))
    return items[:split], items[split:]


class DreemDatasets:
    """
    Context Manager for DreepDataset to open and close datasets properly
    """

    def __init__(self, data_path: str, target_path: str = None, keep_datasets: List[str] = None,
                 split_train_val: float = 0.8, seed: float = None, balance_data=True, size=None,
                 transforms: dict = None, transforms_val: dict = None, verbose=True):
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
            seed: Seed to use.
            size: Size of the dataset to keep
            balance_data: if true, limit dataset to 1400 samples
            transforms: Dict of transformations to apply to the element. The dict is indexed by the name of the dataset
                and has a function as value that takes in a corresponding signal and has to return the transformed signal.
            transforms_val: Dict of transformations to apply to the element of val set. If None, uses the same as train set.
        """
        self.seed = seed
        self.data_path = data_path
        self.target_path = target_path
        self.keep_datasets = keep_datasets
        self.split_train_val = split_train_val
        self.df = pd.read_csv(target_path)
        self.index_labels = {i: self.df.index[self.df.sleep_stage == i].tolist() for i in range(5)}
        self.balance_data = balance_data
        self.size = size
        self.transforms = transforms if transforms is not None else {}
        self.transforms_val = transforms_val if transforms_val is not None else self.transforms
        self.verbose = verbose

    def get(self):
        # Start by initialising the first one to get the size of the dataset
        self.train = DreemDataset(self.data_path, self.target_path, self.keep_datasets,
                                  transforms=self.transforms, verbose=self.verbose).init()
        # Get the split
        if self.balance_data:
            keys_train, keys_val = split_train_validation(len(self.index_labels[1]), self.split_train_val, self.seed,
                                                          self.size, index_labels=self.index_labels)
        else:
            keys_train, keys_val = split_train_validation(len(self.train), self.split_train_val, self.seed, self.size)
        self.train.set_keys_to_keep(keys_train)
        self.val = DreemDataset(self.data_path, self.target_path, self.keep_datasets, keys_val,
                                transforms=self.transforms_val, verbose=self.verbose).init()
        return self.train, self.val

    def __enter__(self):
        return self.get()

    def close(self):
        self.train.close()
        self.val.close()

    def __exit__(self, *args):
        self.close()


class DreemDataset:
    targets = None
    length = None
    h5_datasets = {}
    datasets = None

    def __init__(self, data_path: str, target_path: str = None, keep_datasets: List[str] = None,
                 keys_to_keep: List[int] = None, transforms: dict = None, verbose=True):
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
            transforms: Dict of transformations to apply to the element. The dict is indexed by the name of the dataset
                and has a function as value that takes in a corresponding signal and has to return the transformed signal.
        """
        self.verbose = verbose
        self.keys_to_keep = keys_to_keep
        self.keep_datasets = keep_datasets if keep_datasets is not None else ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4',
                                                                              'eeg_5', 'eeg_6', 'eeg_7',
                                                                              'accelerometer_x', 'accelerometer_y',
                                                                              'accelerometer_z',
                                                                              'pulse_oximeter_infrared']
        self.data_path = data_path
        self.target_path = target_path
        self.transforms = transforms if transforms is not None else {}
        self.separation_50hz_10hz = [['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7'],
                                     ['accelerometer_x', 'accelerometer_y', 'accelerometer_z',
                                      'pulse_oximeter_infrared']]
        self.as_torch_dataset = False

    def init(self):
        self._open_datasets(self.data_path)
        if self.target_path is not None:
            self._load_target(self.target_path)
        self.length = self.length if self.keys_to_keep is None else len(self.keys_to_keep)
        return self

    def torch_dataset(self):
        self.as_torch_dataset = True
        return self

    def set_keys_to_keep(self, keys_to_keep: List[int]):
        self.keys_to_keep = keys_to_keep
        self.length = self.length if keys_to_keep is None else len(keys_to_keep)

    def _open_datasets(self, data_path):
        path = os.path.abspath(os.path.join(os.curdir, data_path))
        self.data = h5py.File(path, 'r')
        for item in self.data:
            if self.keep_datasets is not None and item in self.keep_datasets:
                if self.length is None:
                    self.length = self.data[item].shape[0]
                self.h5_datasets[item] = self.data[item]

    def get_dataset(self, dataset_name, path=None):
        self.v_print("Loading dataset", dataset_name, "...")
        if path is not None:
            return np.load(path + "/" + dataset_name + ".npy")
        else:
            dataset = self.h5_datasets[dataset_name]
            dataset = dataset[:][self.keys_to_keep]  # Only keep the keys_to_keep elements
            if dataset_name in self.transforms.keys():
                self.v_print("Apply transformations...")
                dataset = self.transforms[dataset_name](dataset)
                self.v_print("Applied.")
            return dataset

    def load_data(self, path=None):
        self.v_print("Loading data in memory...")
        self.v_print(len(self), "in", len(self.keep_datasets), "datasets to load")
        self.datasets = {}
        for dataset_name in self.h5_datasets.keys():
            self.datasets[dataset_name] = self.get_dataset(dataset_name, path)
        self.targets = np.array([self.targets[i] for i in self.keys_to_keep])
        if path is not None:
            self.load_targets(path + "/" + "targets.npy")
        self.v_print("Done.")

    def load_targets(self, filename):
        self.targets = np.load(filename)

    def save_data(self, path):
        self.v_print("Saving into", path, "...")
        if not os.path.exists(path):
            os.makedirs(path)
        for dataset_name in self.h5_datasets.keys():
            dataset = self.get_dataset(dataset_name)  # Force not loading from path
            np.save(path + "/" + dataset_name + ".npy", dataset)
        self.targets = np.array([self.targets[i] for i in self.keys_to_keep])
        self.save_targets(path + "/" + "targets.npy")
        self.v_print("Saved.")

    def save_targets(self, filename):
        np.save(filename, self.targets)

    def _load_target(self, target_path):
        path = os.path.abspath(os.path.join(os.curdir, target_path))
        with open(path, 'r') as f:
            reader = csv.reader(f)
            self.targets = {int(i): int(j) for i, j in list(reader)[1:]}

    def close(self):
        """
        Closes the dataset
        """
        self.data.close()

    def __enter__(self):
        return self.init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        # item = item if self.keys_to_keep is None else self.keys_to_keep[item]
        data_50hz = []
        data_10hz = []
        if self.datasets is None:
            self.load_data()
        for dataset_name, dataset in self.datasets.items():
            data = dataset[item]
            # data = data if dataset_name not in self.transforms.keys() else self.transforms[dataset_name](data)
            if dataset_name in self.separation_50hz_10hz[0]:
                data_50hz.append(data)
            else:
                data_10hz.append(data)
        data_50hz = np.array(data_50hz)
        data_10hz = np.array(data_10hz)
        targets = self.targets[item]
        if not data_10hz:
            data_10hz = np.array([[0]])
        if self.as_torch_dataset:
            data_50hz = torch.tensor(data_50hz)
            data_10hz = torch.tensor(data_10hz)
            targets = torch.tensor(targets, dtype=torch.int64)
        return (data_50hz, data_10hz, targets) if self.targets is not None else (data_50hz, data_10hz)

    def __len__(self):
        return self.length

    def v_print(self, *text):
        """
        Verbose print.
        Only prints if verbose is activated
        """
        if self.verbose:
            print(*text)
