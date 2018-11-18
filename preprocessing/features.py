import numpy as np
import matplotlib.pyplot as plt

__all__ = ["ExtractFeatures"]


class ExtractFeatures:
    def __init__(self, features=None, window=100, sampling_freq=50):
        """

        Args:
            window: size of the sliding window
            sampling_freq:
            features: list of features to use. If None, all of them. Available features:
                - min
                - max
                - frequency
                - energy
                - mmd: maximum-minimum distance
                - esis: EnergySis
        """
        self.window = window
        self.samplig_freq = sampling_freq
        available_features = {
            'min': lambda s: min(s),
            'max': lambda s: max(s),
            'frequency': self.get_frequency,
            'energy': self.get_energy,
            'mmd': self.get_mmd,
            'esis': self.get_esis
        }
        features = features if features is not None else available_features.keys()
        self.features = [available_features[feature] for feature in features]

    def __call__(self, signal):
        features = None
        if len(signal.shape) == 1:
            features = self.get_features(signal)
        elif len(signal.shape) == 2:  # decomposed in bands
            features = []
            for band in range(signal.shape[0]):
                features.append(self.get_features(signal[band]))
            features = np.array(features)
            features = features.reshape((features.shape[0] * features.shape[1]))
        return features

    def get_features(self, signal):
        features = [feature(signal) for feature in self.features]
        return np.array(features)

    def get_frequency(self, signal):
        spectrum = np.fft.fft(signal) / self.samplig_freq
        freqs = np.fft.fftfreq(signal.shape[0])
        threshold = 0.99 * max(abs(spectrum))
        mask = abs(spectrum) >= threshold
        # spectrum = spectrum[mask]
        freqs = freqs[mask]
        mask_pos = freqs >= 0
        return 0 if len(abs(freqs[mask_pos])) == 0 else abs(freqs[mask_pos])[0]

    def get_energy(self, signal):
        return sum(np.power(signal, 2)) / self.samplig_freq

    def get_mmd(self, signal):
        """
        Feature from "Sleep Stage Classification Using EEG Signal Analysis:
            A Comprehensive Survey and New Investigation", Aboalayon et al.
        """
        range_signal = range(0, len(signal), self.window)
        d = np.zeros(len(list(range_signal)))
        for i, k in enumerate(range_signal):
            sub_signal = signal[k:k + self.window]
            arg_min = np.argmin(sub_signal)
            arg_max = np.argmax(sub_signal)
            d[i] = np.sqrt((max(sub_signal) - min(sub_signal)) ** 2 + (arg_max - arg_min) ** 2)
        return d.sum()

    def get_esis(self, signal):
        """
        Feature from "Sleep Stage Classification Using EEG Signal Analysis:
            A Comprehensive Survey and New Investigation", Aboalayon et al.
        """
        return self.get_energy(signal) * self.get_frequency(signal) * self.window

    def get_inflexion(self, signal):
        h = 1 / self.samplig_freq
        num_inflexion = 0
        for k in range(1, len(signal) - 1):
            d = (signal[k - 1] + signal[k + 1] - 2 * signal[k]) / (2 * h)
            if abs(d) < 0.01:
                num_inflexion += 1
        return num_inflexion
