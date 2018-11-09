import torch
from tools import Parser
from tools.data import DreemDataset


# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

train_set = DreemDataset('dataset/train.h5', 'dataset/train_y.csv')

print(train_set[0])