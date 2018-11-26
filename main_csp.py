#!/usr/bin/env python3

''' 
Model for common spatial pattern (CSP) feature calculation and classification for EEG data
'''

import time

import numpy as np
from tools.csp import generate_projection, generate_eye, extract_feature
from tools.filters import load_filterbank
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"


class CSP_Model:
    def __init__(self):
        self.crossvalidation = False
        self.data_path = 'dataset/'
        self.svm_kernel = 'RF'  # 'sigmoid'#'linear' # 'sigmoid', 'rbf', 'poly'
        self.svm_c = 0.1  # 0.05 for linear, 20 for rbf, poly: 0.1
        self.useCSP = True
        self.NO_splits = 5  # number of folds in cross validation
        self.fs = 50.  # sampling frequency
        self.NO_channels = 7  # number of EEG channels
        self.NO_subjects = 1
        self.NO_csp = 24  # Total number of CSP feature per band and timewindow
        self.bw = np.array([2, 4, 8, 16, 22])  # bandwidth of filtered signals
        # self.bw = np.array([1,2,4,8,16,32])
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
                                     [0, 10],
                                     [0, 30]]) * self.fs  # time windows in [s] x fs for using as a feature

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

    def run_csp(self):

        ################################ Training ############################################################################
        start_train = time.time()
        # 1. Apply CSP to bands to get spatial filter
        if self.useCSP:
            w = generate_projection(self.train_data, self.train_label, self.NO_csp, self.filter_bank, self.time_windows, NO_classes=5

                                    )
        else:
            w = generate_eye(self.train_data, self.train_label, self.filter_bank, self.time_windows)



        # 2. Extract features for training
        feature_mat = extract_feature(self.train_data, w, self.filter_bank, self.time_windows)
        #np.save("./dataset/all_eegs_balanced/features_all_csp.npy", feature_mat)
        #feature_mat = np.load("./dataset/all_eegs_balanced/features_all_csp.npy")

        # 3. Stage Train SVM Model
        # 2. Train NN
        # 3. Stage Train SVM Model
        # 2. Train SVM Model
        if self.svm_kernel == 'linear':
            clf = LinearSVC(C=self.svm_c, intercept_scaling=1, loss='hinge', max_iter=10000, multi_class='ovr',
                            penalty='l2', random_state=1, tol=0.00001)

        elif self.svm_kernel == 'RF':

            clf = RandomForestClassifier(n_estimators=100, random_state=0)


        else:
            clf = SVC(self.svm_c, self.svm_kernel, degree=10, gamma='auto', coef0=0.0, tol=0.001, cache_size=10000,
                      max_iter=-1, decision_function_shape='ovr')
        clf.fit(feature_mat, self.train_label)

        end_train = time.time()
        self.train_time += end_train - start_train
        self.train_trials += len(self.train_label)

        ################################# Evaluation ###################################################
        start_eval = time.time()
        eval_feature_mat = extract_feature(self.eval_data, w, self.filter_bank, self.time_windows)
        #np.save("./dataset/all_eegs_balanced/features_5000_csp_test.npy", eval_feature_mat)
        #eval_feature_mat = np.load("./dataset/all_eegs_balanced/features_5000_csp_val.npy")

        #success_rate = clf.score(eval_feature_mat, self.eval_label)

        end_eval = time.time()

        # print("Time for one Evaluation " + str((end_eval-start_eval)/len(self.eval_label)) )

        self.eval_time += end_eval - start_eval
        #self.eval_trials += len(self.eval_label)

        labels_pred = clf.predict(eval_feature_mat)

        import pandas as pd
        df = pd.DataFrame(labels_pred)
        #df.to_csv("labels_pred_t.csv")


        CM = confusion_matrix(self.eval_label, labels_pred)
        Acc = accuracy_score(self.eval_label, labels_pred)
        F1 = f1_score(self.eval_label, labels_pred, average='macro')


        print(CM, Acc, F1)


        return (F1)

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
            self.train_data, self.train_label = self.get_data(one_vs_all = False)
            self.eval_data, self.eval_label = self.get_data(train = False, one_vs_all = False)


    def get_data(self, train= True,  one_vs_all = False, limit_300= False):
        if not one_vs_all:
            if train:
                X = np.zeros((7, 500, 1500))
                for i in range(7):
                    X[i] = np.load("dataset/all_eegs/train/eeg_" + str(i + 1) + ".npy")[:500]
                Y = np.load("dataset/all_eegs/train/targets.npy")[:500]
                X = X.transpose((1, 0, 2))
            else:
                X = np.zeros((7, 500, 1500))
                for i in range(7):
                    X[i] = np.load("dataset/all_eegs/val/eeg_" + str(i + 1) + ".npy")[:500]
                Y = np.load("dataset/all_eegs/val/targets.npy")[:500]
                X = X.transpose((1, 0, 2))
        if one_vs_all:
            if train:
                X = np.zeros((7, 30631, 1500))
                for i in range(7):
                    X[i] = np.load("dataset/all_eegs/train/eeg_" + str(i + 1) + ".npy")[:30631]
                Y = np.load("dataset/all_eegs/train/targets.npy")[:30631]
                X = X.transpose((1, 0, 2))
            else:
                X = np.zeros((7, 7658, 1500))
                for i in range(7):
                    X[i] = np.load("dataset/all_eegs/val/eeg_" + str(i + 1) + ".npy")[:7658]
                Y = np.load("dataset/all_eegs/val/targets.npy")[:7658]
                X = X.transpose((1, 0, 2))
            Y[Y > 2] = 0
            Y[Y < 2] = 0
            Y[Y == 2] = 1
        if limit_300:
            X = X[:300]
            Y = Y[:300]
        return(X, Y)



def main():
    model = CSP_Model()

    print("Number of used features: " + str(model.NO_features))

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
                success_sub_sum += model.run_csp()
                print(success_sub_sum / (model.split + 1))
            # average over all splits
            success_rate = success_sub_sum / model.NO_splits

        else:
            # load Eval data
            model.load_data()
            success_rate = model.run_csp()

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
