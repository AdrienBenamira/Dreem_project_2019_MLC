from preprocessing.signals import *
from preprocessing.features import *


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations
        self.name = "_".join([transformation.name for transformation in self.transformations])

    def __call__(self, signal, target):
        for transfo in self.transformations:
            signal = transfo(signal, target)
        return signal
