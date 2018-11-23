import torch.optim as optim
import torch.utils.data
from tools import Parser, SimpleTrainer as Trainer
from tools.data import DreemDatasets
from tools import show_curves
from models import CNN, MLPClassifier, SimpleCNN
from preprocessing import Compose, ExtractBands, ExtractFeatures, ExtractSpectrum
import matplotlib.pyplot as plt

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

use_datasets = ['eeg_1', 'eeg_4', 'eeg_5', 'eeg_6']

# Fist test, only 2 networks, 1 for eeg, 1 for accelerometers and pulse
number_groups = 4
out_features = 100
in_channel_50hz = 4 * 3  # 4 bands and 3 eeg curves
in_channel_10hz = 2  # 1 accelerometer and 1 pulse
channels = [8, 16, 32, 16]
# in_channels are 4s because 4 sampled at 50Hz and 4 sampled at 10Hz
# model_50hz = CNN(in_features=1500, out_features=out_features * in_channel_50hz, in_channels=in_channel_50hz,
#                  number_groups=number_groups, size_groups=1, hidden_channels=channels)
# model_10hz = CNN(in_features=300, out_features=out_features * in_channel_10hz, in_channels=in_channel_10hz,
#                  number_groups=number_groups, size_groups=1, hidden_channels=channels)

# in features: 3 in dataset 50hz, 4bands each with 6 features. 2 in dataset 10hz with 1 feature each
# classifier = MLPClassifier(in_features=3*4*2+2, out_features=5)
model = SimpleCNN()

if use_cuda:
    print('Use GPU')
    model.cuda()
    # model_10hz.cuda()
    # model_50hz.cuda()
    # classifier.cuda()

dataset_transforms = {
    "eeg_1": Compose([ExtractBands(), ExtractSpectrum(window=50)]),
    "eeg_2": Compose([ExtractBands(), ExtractSpectrum(window=50)]),
    "eeg_3": Compose([ExtractBands(), ExtractSpectrum(window=50)]),
    "eeg_4": Compose([ExtractBands(), ExtractSpectrum(window=50)]),
    "eeg_5": ExtractFeatures(['esis'], bands='*'),
    "accelerometer_x": ExtractFeatures(['energy']),
    "accelerometer_y": ExtractFeatures(['energy']),
    "accelerometer_z": ExtractFeatures(['energy']),
    "pulse_oximeter_infrared": ExtractFeatures(['frequency'])
}


def trainer_transform(data_50hz, data_10hz):
    """
    Trainer transform to set all eeg and bands in same dimension
    """
    data_50hz = data_50hz.view(data_50hz.size(0), -1, 30, 50)
    return data_50hz, None


# Use context manager to close the datasets when we're finished!
with DreemDatasets('dataset/train.h5', 'dataset/train_y.csv',
                   keep_datasets=use_datasets, split_train_val=0.8, seed=args.seed,
                   size=5000) as (train_set, val_set):
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=1)
    train_set.save_data('dataset/test')
    # train_set.load_data('dataset/dataset_all_eeg.npy')

    data, _, targets = train_set[0]
    print(data.shape)
    # train_set.sa

    # for k in range(3):
    #     data_50hz, data_10hz, target = train_set[k]
    #     print(data_50hz)
    #     print(data_10hz)
    #     show_curves(data_50hz, data_10hz, target)
    # plt.show()

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.lr,
    #                       momentum=args.momentum)
    # trainer = Trainer(train_loader, test_loader, optimizer, classifier=model,
    #                   log_every=10, save_folder='builds', transform=trainer_transform)
    # trainer.train(args.epochs)
