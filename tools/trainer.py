from datetime import datetime
import os

import torch

__all__ = ['Trainer']

use_cuda = torch.cuda.is_available()


class Trainer:
    def __init__(self, train_loader, val_loader, optimizer, model_50hz, model_10hz, classifier, criterion,
                 log_every: int = 50, save_folder: str = None):
        """
        Trainer class
        Args:
            train_loader: training loader
            val_loader: validation loader
            optimizer: optimizer
            model_50hz:
            model_10hz:
            classifier:
            criterion: Loss criterion
            log_every: Print log every batch
            save_folder: folder to save the learned models
        """
        self.save_folder = save_folder
        self.log_every = log_every
        self.criterion = criterion
        self.classifier = classifier
        self.model_10hz = model_10hz
        self.model_50hz = model_50hz
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader

    def step_train(self, epoch):
        self.model_50hz.train()
        self.model_10hz.train()

        for batch_id, (data_50hz, data_10hz, target) in enumerate(self.train_loader):
            if use_cuda:
                data_50hz, data_10hz, target = data_50hz.cuda(), data_10hz.cuda(), target.cuda()
            self.optimizer.zero_grad()
            out_50hz = self.model_50hz(data_50hz)
            out_10hz = self.model_10hz(data_10hz)
            out = self.classifier(torch.cat((out_50hz, out_10hz), dim=-1))
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            if batch_id % self.log_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data_50hz), len(self.train_loader.dataset),
                           100. * batch_id / len(self.train_loader.dataset), loss.data.item()))

    def step_val(self, epoch):
        self.model_50hz.eval()
        self.model_10hz.eval()
        validation_loss = 0
        correct = 0
        for batch_id, (data_50hz, data_10hz, target) in enumerate(self.train_loader):
            if use_cuda:
                data_50hz, data_10hz, target = data_50hz.cuda(), data_10hz.cuda(), target.cuda()
            out_50hz = self.model_50hz(data_50hz)
            out_10hz = self.model_10hz(data_10hz)
            out = self.classifier(torch.cat((out_50hz, out_10hz), dim=-1))
            validation_loss += self.criterion(out, target).data.item()
            # get the index of the max log-probability
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(self.val_loader)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(self.val_loader),
            100. * correct / len(self.val_loader)))

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
