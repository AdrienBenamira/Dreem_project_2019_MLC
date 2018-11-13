import torch
import torch.utils.data
from tools import Parser
from tools.data import DreemDatasets

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

# Use context manager to close the datasets when we're finished!
with DreemDatasets('dataset/train.h5', 'dataset/train_y.csv', keep_datasets=['eeg_2', 'eeg_4'],
                   split_train_val=0.8) as (train_set, val_set):
    train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
