import argparse

__all__ = ['Parser']


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Dreem training')
        self.parser.add_argument('-b', '--batch-size', type=int, default=4,
                                 help="Batch size")
        self.parser.add_argument('-s', '--seed', type=float, default=1,
                                 help="Seed")

    def parse(self):
        return self.parser.parse_args()
