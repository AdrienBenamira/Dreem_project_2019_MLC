from typing import Callable, Tuple
from torch import Tensor
from datetime import datetime
import os

import torch

__all__ = ['Trainer']

use_cuda = torch.cuda.is_available()


class GenericTrainer:
    def __init__(self, train_loader, val_loader, optimizer, model_50hz, model_10hz, classifier,
                 log_every: int = 50, save_folder: str = None,
                 transform: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None):
        """
        Trainer class
        Args:
            train_loader: training loader
            val_loader: validation loader
            optimizer: optimizer
            model_50hz:
            model_10hz:
            classifier:
            log_every: Print log every batch
            save_folder: folder to save the learned models
            transform: Transforms the data out of the model_50hz and model_10hz. Callable taking data_50hz and
                data_10hz s input and returns the transformed data_50Hz and data_10Hz.
        """
        self.save_folder = save_folder
        self.log_every = log_every
        self.classifier = classifier
        self.model_10hz = model_10hz
        self.model_50hz = model_50hz
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.transform = (lambda x, y: (x, y)) if transform is None else transform

    def step_train(self, epoch):
        self.model_50hz.train()
        self.model_10hz.train()

        for batch_id, (data_50hz, data_10hz, target) in enumerate(self.train_loader):
            if use_cuda:
                data_50hz, data_10hz, target = data_50hz.cuda(), data_10hz.cuda(), target.cuda()
            self.optimizer.zero_grad()
            data_50hz, data_10hz = self.transform(data_50hz, data_10hz)
            loss, _ = self.forward(data_50hz, data_10hz, target)
            loss.backward()
            self.optimizer.step()
            if batch_id % self.log_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data_50hz), len(self.train_loader.dataset),
                           100. * batch_id * target.size(0) / len(self.train_loader.dataset), loss.data.item()))

    def step_val(self, epoch):
        self.model_50hz.eval()
        self.model_10hz.eval()
        validation_loss = 0
        correct = 0
        batch_size = None
        for batch_id, (data_50hz, data_10hz, target) in enumerate(self.val_loader):
            if batch_size is None:
                batch_size = target.size(0)
            if use_cuda:
                data_50hz, data_10hz, target = data_50hz.cuda(), data_10hz.cuda(), target.cuda()
            data_50hz, data_10hz = self.transform(data_50hz, data_10hz)
            loss, out = self.forward(data_50hz, data_10hz, target)
            validation_loss += loss.data.item()
            # get the index of the max log-probability
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(self.val_loader.dataset) / batch_size
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(self.val_loader.dataset),
            100. * correct / len(self.val_loader.dataset)))

    def train(self, n_epochs: int):
        """
        Args:
            n_epochs: Number of epochs
        """
        if self.save_folder is not None:
            path = self.save_folder + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            os.mkdir(path)
        for epoch in range(1, n_epochs + 1):
            self.step_train(epoch)
            self.step_val(epoch)
            if self.save_folder is not None:
                model_file = path + '/model_50hz.pth'
                torch.save(self.model_50hz.state_dict(), model_file)
                model_file = path + '/model_10hz.pth'
                torch.save(self.model_10hz.state_dict(), model_file)
                model_file = path + '/classifier.pth'
                torch.save(self.classifier.state_dict(), model_file)
            print('\nSaved models in ' + path + '.')

    def forward(self, data_50hz, data_10hz, target):
        raise NotImplementedError


class Trainer(GenericTrainer):
    def forward(self, data_50hz, data_10hz, target):
        out_50hz = self.model_50hz(data_50hz)
        out_10hz = self.model_10hz(data_10hz)
        out = self.classifier(torch.cat((out_50hz, out_10hz), dim=-1))
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(out, target)
        return loss, out

