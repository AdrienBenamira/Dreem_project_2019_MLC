import numpy as np
import matplotlib.pyplot as plt

__all__ = ["GenomicFeatures"]


class GenomicFeatures:
    number_features = 3

    def __init__(self, window=100, sampling_freq=50):
        """

        Args:
            window: size of the sliding window
        """
        self.window = window
        self.samplig_freq = sampling_freq

    def __call__(self, signal):
        features = None
        if len(signal.shape) == 1:
            features = self.compute(signal)
        elif len(signal.shape) == 2:  # decomposed in bands
            features = []
            for band in range(signal.shape[0]):
                features.append(self.compute(signal[band]))
            features = np.array(features)
            features = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
        return features

    def compute(self, signal):
        range_signal = range(0, len(signal), self.window)
        features = np.zeros((self.number_features, len(list(range_signal))))
        for i, k in enumerate(range_signal):
            sub_signal = signal[k: k + self.window]
            features[:, i] = self.get_features(sub_signal)
        return features

    def get_features(self, signal):
        freqs = self.get_frequency(signal)
        features = [
            0 if len(freqs) == 0 else freqs[0],
            min(signal),
            max(signal)
            # self.inflexion(signal)
        ]
        return np.array(features)

    def get_frequency(self, signal):
        spectrum = np.fft.fft(signal) / self.samplig_freq
        freqs = np.fft.fftfreq(self.window)
        threshold = 0.9 * max(abs(spectrum))
        mask = abs(spectrum) > threshold
        spectrum = spectrum[mask]
        freqs = freqs[mask]
        mask_pos = freqs >= 0
        return abs(spectrum[mask_pos])

    def inflexion(self, signal):
        h = 1 / self.samplig_freq
        num_inflexion = 0
        for k in range(1, len(signal)-1):
            d = (signal[k-1] + signal[k+1] - 2*signal[k]) / (2*h)
            if abs(d) < 0.01:
                num_inflexion += 1
        return num_inflexion

