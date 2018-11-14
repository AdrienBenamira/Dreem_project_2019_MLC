import torch
import torch.utils.data
from tools import Parser
from tools.data import DreemDatasets
from models import CNN, MLPClassifier

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

use_datasets = ['eeg_1', 'eeg_3', 'eeg_4', 'eeg_7', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
                'pulse_oximeter_infrared']

# Fist test, only 2 networks, 1 for eeg, 1 for accelerometers and pulse
number_groups = 4
model_50hz = CNN(in_features=1500, out_features=100, number_groups=number_groups, size_groups=2,
                 hidden_channels=[4] * number_groups)  # hidden_channels of 4 because 4 channels eeg
model_10hz = CNN(in_features=300, out_features=100, number_groups=number_groups, size_groups=2,
                 hidden_channels=[4] * number_groups)  # hidden_channels of 4 because 4 channels sampled at 10Hz
classifier = MLPClassifier(in_features=100 * len(use_datasets), out_features=5)

# Use context manager to close the datasets when we're finished!
with DreemDatasets('dataset/train.h5', 'dataset/train_y.csv', keep_datasets=use_datasets,
                   split_train_val=0.8, seed=args.seed) as (train_set, val_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
    for data_50hz, data_10hz, target in train_loader:
        print(data_50hz.size())
        print(data_10hz.size())
        break
