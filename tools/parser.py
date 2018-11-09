import argparse

__all__ = ['Parser']


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='RecVis A3 training script')
        self.parser.add_argument('--test', type=str, default='Test',
                                 help="This is an argument")

    def parse(self):
        return self.parser.parse_args()
