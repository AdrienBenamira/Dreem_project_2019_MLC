import torch
import torch.optim as optim
import torch.utils.data
from tools import Parser, Trainer
from tools.data import DreemDatasets
from models import CNN, MLPClassifier

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

use_datasets = ['eeg_1', 'eeg_4', 'accelerometer_x', 'pulse_oximeter_infrared']

# Fist test, only 2 networks, 1 for eeg, 1 for accelerometers and pulse
number_groups = 4
out_features = 200
# in_channels are 4s because 4 sampled at 50Hz and 4 sampled at 10Hz
model_50hz = CNN(in_features=1500, out_features=out_features, in_channels=2, number_groups=number_groups, size_groups=1)
model_10hz = CNN(in_features=300, out_features=out_features, in_channels=2, number_groups=number_groups, size_groups=1)
classifier = MLPClassifier(in_features=out_features * 2, out_features=5)

if use_cuda:
    print('Use GPU')
    model_10hz.cuda()
    model_50hz.cuda()
    classifier.cuda()

# Use context manager to close the datasets when we're finished!
with DreemDatasets('dataset/train.h5', 'dataset/train_y.csv', keep_datasets=use_datasets,
                   split_train_val=0.8, seed=args.seed) as (train_set, val_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
    criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    optimizer = optim.SGD([{'params': model_50hz.parameters()},
                           {'params': model_10hz.parameters()},
                           {'params': classifier.parameters()}],
                          lr=args.lr,
                          momentum=args.momentum)

    trainer = Trainer(train_loader, test_loader, optimizer, model_50hz, model_10hz, classifier, criterion,
                      log_every=10, save_folder='builds')

    trainer.train(args.epochs)
