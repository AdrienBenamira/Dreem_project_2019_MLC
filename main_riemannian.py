#!/usr/bin/env python3

''' 
Model for Riemannian feature calculation and classification for EEG data
'''

import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC

from models.filters import load_filterbank
from models.riemannian_multiscale import riemannian_multiscale

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression




__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"
import os


class Riemannian_Model:
    def __init__(self):
        self.crossvalidation = False
        self.data_path = 'dataset/'
        self.svm_kernel = 'RF'  # 'sigmoid'#'linear' # 'sigmoid', 'rbf',
        self.svm_c = 0.1  # for linear 0.1 (inverse),
        self.NO_splits = 5  # number of folds in cross validation
        self.fs = 50.  # sampling frequency
        self.NO_channels = 7  # number of EEG channels
        self.NO_subjects = 1
        self.NO_riem = int(
            self.NO_channels * (self.NO_channels + 1) / 2)  # Total number of CSP feature per band and timewindow
        # self.bw = np.array([2,4,8,16,32]) # bandwidth of filtered signals
        self.bw = np.array([2, 4, 8, 16, 22])
        self.ftype = 'butter'  # 'fir', 'butter'
        self.forder = 2  # 4
        self.filter_bank = load_filterbank(self.bw, self.fs, order=self.forder, max_freq=23,
                                           ftype=self.ftype)  # get filterbank coeffs
        time_windows_flt = np.array([[0, 30],
                                     [15, 30],
                                     [10, 25],
                                     [5, 20],
                                     [0, 15],
                                     [15, 25],
                                     [10, 20],
                                     [5, 15],
                                     [0, 10]]) * self.fs
        self.time_windows = time_windows_flt.astype(int)
        # restrict time windows and frequency bands
        self.time_windows = self.time_windows[2:3]  # use only largest timewindow
        #self.f_bands_nom = self.f_bands_nom[18:27] # use only 4Hz-32Hz bands
        self.rho = 0.1
        self.NO_bands = self.filter_bank.shape[0]
        self.NO_time_windows = self.time_windows.shape[0]
        self.NO_features = self.NO_riem * self.NO_bands * self.NO_time_windows
        self.riem_opt = "Riemann"  # {"Riemann","Riemann_Euclid","Whitened_Euclid","No_Adaptation"}
        # time measurements
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0

    def run_riemannian(self):

        ################################ Training ############################################################################
        start_train = time.time()

        # 1. calculate features and mean covariance for training
        riemann = riemannian_multiscale(self.filter_bank, self.time_windows, riem_opt=self.riem_opt, rho=self.rho,
                                        vectorized=True)
        train_feat = riemann.fit(self.train_data)

        # 2. Train SVM Model
        if self.svm_kernel == 'linear':
            clf = LinearSVC(C=self.svm_c, intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
                            penalty='l2', random_state=1, tol=0.00001)

        elif self.svm_kernel == 'RF':

            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            #LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)

            print("ok")

        else:
            clf = SVC(self.svm_c, self.svm_kernel, degree=10, gamma='auto', coef0=0.0, tol=0.001, cache_size=10000,
                      max_iter=-1, decision_function_shape='ovr')

        clf.fit(train_feat, self.train_label)

        end_train = time.time()
        self.train_time += end_train - start_train
        self.train_trials += len(self.train_label)

        ################################# Evaluation ###################################################
        start_eval = time.time()

        eval_feat = riemann.features(self.eval_data)

        success_rate = clf.score(eval_feat, self.eval_label)

        end_eval = time.time()

        # print("Time for one Evaluation " + str((end_eval-start_eval)/len(self.eval_label)) )

        self.eval_time += end_eval - start_eval
        self.eval_trials += len(self.eval_label)

        return success_rate

    def load_data(self):
        if self.crossvalidation:
            data, label = self.get_data(self.subject, True, self.data_path)
            kf = KFold(n_splits=self.NO_splits)
            split = 0
            for train_index, test_index in kf.split(data):
                if self.split == split:
                    self.train_data = data[train_index]
                    self.train_label = label[train_index]
                    self.eval_data = data[test_index]
                    self.eval_label = label[test_index]
                split += 1
        else:
            self.train_data, self.train_label = self.get_data( True, self.data_path)
            self.eval_data, self.eval_label = self.get_data( False, self.data_path)
            print(self.train_data.shape, self.train_label.shape)


    def get_data(self, train= True, PATH = "dataset/all_eegs/train/", one_vs_all = True, limit_300= True):

        if train:
            X = np.zeros((7, 5412, 1500))
            for i in range(7):
                X[i] = np.load("dataset/all_eegs/train/eeg_" + str(i + 1) + ".npy")
            Y = np.load("dataset/all_eegs/train/targets.npy")
            X = X.reshape(5412, 7, 1500)
            #X = np.load("./dataset/X_features_train_equilibrate_1500.npy")
            #Y = np.load("./dataset/Y_labels_train_equilibrate.npy")



        else:
            X = np.zeros((7, 1353, 1500))
            for i in range(7):
                X[i] = np.load("dataset/all_eegs/val/eeg_" + str(i + 1) + ".npy")
            Y = np.load("dataset/all_eegs/val/targets.npy")
            X = X.reshape(1353, 7, 1500)
            #X = np.load("./dataset/X_features_val_equilibrate_1500.npy")
            #Y = np.load("./dataset/Y_labels_val_equilibrate.npy")

        if one_vs_all:
            Y[Y == 2] = 1
            Y[Y > 2] = 0
            Y[Y < 2] = 0

        if limit_300:
            X = X[:300]
            Y = Y[:300]

        return(X, Y)


def main():
    model = Riemannian_Model()

    print("Number of used features: " + str(model.NO_features))

    print(model.riem_opt)

    # success rate sum over all subjects
    success_tot_sum = 0

    if model.crossvalidation:
        print("Cross validation run")
    else:
        print("Test data set")

    start = time.time()

    # Go through all subjects
    for model.subject in range(1, model.NO_subjects + 1):

        # print("Subject" + str(model.subject)+":")


        if model.crossvalidation:
            success_sub_sum = 0

            for model.split in range(model.NO_splits):
                model.load_data()
                success_sub_sum += model.run_riemannian()

            # average over all splits
            success_rate = success_sub_sum / model.NO_splits



        else:
            # load Eval data
            model.load_data()
            success_rate = model.run_riemannian()

        print(success_rate)
        success_tot_sum += success_rate



    # Average success rate over all subjects
    print("Average success rate: " + str(success_tot_sum / model.NO_subjects))

    print("Training average time: " + str(model.train_time / model.NO_subjects))
    print("Evaluation average time: " + str(model.eval_time / model.NO_subjects))

    end = time.time()

    print("Time elapsed [s] " + str(end - start))


if __name__ == '__main__':
    main()
