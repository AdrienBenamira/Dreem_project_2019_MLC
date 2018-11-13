import argparse

__all__ = ['Parser']


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Dreem training')
        self.parser.add_argument('--batch-size', type=int, default=64,
                                 help="Batch size")

    def parse(self):
        return self.parser.parse_args()
