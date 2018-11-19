from preprocessing.signals import *
from preprocessing.features import *


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, signal):
        for transfo in self.transformations:
            signal = transfo(signal)
        return signal
