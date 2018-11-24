# import self defined functions
from preprocessing.csp import *
from preprocessing.eig import *
from preprocessing.filters import *


class CSP_Features:

    def __init__(self):
        """

        """
        self.crossvalidation = False
        self.data_path = 'dataset/'
        self.svm_kernel = 'linear'  # 'sigmoid'#'linear' # 'sigmoid', 'rbf', 'poly'
        self.svm_c = 0.1  # 0.05 for linear, 20 for rbf, poly: 0.1
        self.useCSP = True
        self.NO_splits = 5  # number of folds in cross validation
        self.fs = 50  # sampling frequency
        self.NO_channels = 7  # number of EEG channels
        self.NO_subjects = 1
        self.NO_csp = 24  # Total number of CSP feature per band and timewindow
        self.bw = np.array([4, 8, 13, 22])  # bandwidth of filtered signals
        # self.bw = np.array([1,2,4,8,16,32])
        self.ftype = 'butter'  # 'fir', 'butter'
        self.forder = 2  # 4
        self.filter_bank = load_filterbank(self.bw, self.fs, order=self.forder, max_freq=22,
                                           ftype=self.ftype)  # get filterbank coeffs
        time_windows_flt = np.array([[0, 30],
                                     [14, 60],
                                     [10, 26],
                                     [6, 22],
                                     [2, 18],
                                     [0, 8],
                                     [4, 12],
                                     [8, 16],
                                     [12, 20],
                                     [16, 24],
                                     [20, 28],
                                     [24, 30]]) * self.fs  # time windows in [s] x fs for using as a feature

        self.time_windows = time_windows_flt.astype(int)
        # restrict time windows and frequency bands
        # self.time_windows = self.time_windows[10] # use only largest timewindow
        # self.filter_bank = self.filter_bank[18:27] # use only 4Hz bands

        self.NO_bands = self.filter_bank.shape[0]
        self.NO_time_windows = int(self.time_windows.size / 2)
        self.NO_features = self.NO_csp * self.NO_bands * self.NO_time_windows
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0

    def __call__(self, batch_signal):
        """

        :param batch_signal:
        :return:
        """



        print(batch_signal.shape)

        if self.useCSP:
            w = generate_projection(train, label, self.NO_csp, self.filter_bank, self.time_windows,
                                    NO_classes=5)
        else:
            w = generate_eye(train, label, self.filter_bank, self.time_windows, NO_classes=5)


        feature_mat = extract_feature(self.train_data, w, self.filter_bank, self.time_windows)

        return (feature_mat)
