import torch.optim as optim
import torch.utils.data
from tools import Parser, CNNTrainer as Trainer
from tools.data import DreemDatasets
from tools import show_curves
from models import CNN, MLPClassifier, SimpleCNN
from preprocessing import Compose, ExtractBands, ExtractFeatures, ExtractSpectrum
import matplotlib.pyplot as plt

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()

use_datasets = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']

# Fist test, only 2 networks, 1 for eeg, 1 for accelerometers and pulse
number_groups = 4
out_features = 100
in_channel_50hz = 4 * 3  # 4 bands and 3 eeg curves
in_channel_10hz = 2  # 1 accelerometer and 1 pulse
channels = [28, 32, 16, 8]
# in_channels are 4s because 4 sampled at 50Hz and 4 sampled at 10Hz
model_50hz = CNN(in_features=1500, out_features=100, in_channels=7*4,
                 number_groups=number_groups, size_groups=1, hidden_channels=channels)
# model_10hz = CNN(in_features=300, out_features=out_features * in_channel_10hz, in_channels=in_channel_10hz,
#                  number_groups=number_groups, size_groups=1, hidden_channels=channels)

# in features: 3 in dataset 50hz, 4bands each with 6 features. 2 in dataset 10hz with 1 feature each
classifier = MLPClassifier(in_features=100, out_features=5)
# model = SimpleCNN()

if use_cuda:
    print('Use GPU')
    # model.cuda()
    # model_10hz.cuda()
    model_50hz.cuda()
    classifier.cuda()

dataset_transforms = {
    "eeg_1": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_2": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_3": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_4": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_5": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_6": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "eeg_7": Compose([ExtractBands(), ExtractSpectrum(window=100)]),
    "accelerometer_x": ExtractFeatures(['energy']),
    "accelerometer_y": ExtractFeatures(['energy']),
    "accelerometer_z": ExtractFeatures(['energy']),
    "pulse_oximeter_infrared": ExtractFeatures(['frequency'])
}


def trainer_transform(data_50hz, data_10hz):
    """
    Trainer transform to set all eeg and bands in same dimension
    """
    data_50hz = data_50hz.view(-1, 7*4, 1500)
    return data_50hz, None


# Use context manager to close the datasets when we're finished!
with DreemDatasets('dataset/train.h5', 'dataset/train_y.csv',
                   keep_datasets=use_datasets, split_train_val=0.8, seed=args.seed,
                   size=5000, transforms=dataset_transforms) as (train_set, val_set):
    train_set.load_data('dataset/eegs_band_spectrum/train')
    val_set.load_data('dataset/eegs_band_spectrum/val')
    train_loader = torch.utils.data.DataLoader(train_set.torch_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(val_set.torch_dataset(), batch_size=args.batch_size, num_workers=1)

    optimizer = optim.SGD([{'params': model_50hz.parameters()},
                           {'params': classifier.parameters()}],
                          lr=args.lr,
                          momentum=args.momentum)
    trainer = Trainer(train_loader, test_loader, optimizer, model_50hz=model_50hz, classifier=classifier,
                      log_every=10, save_folder='builds', transform=trainer_transform)
    trainer.train(args.epochs)
