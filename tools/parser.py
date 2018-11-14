import argparse

__all__ = ['Parser']


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Dreem training')
        self.parser.add_argument('-b', '--batch-size', type=int, default=4,
                                 help="Batch size")
        self.parser.add_argument('-s', '--seed', type=float, default=1,
                                 help="Seed")
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                                 help='learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')

    def parse(self):
        return self.parser.parse_args()
