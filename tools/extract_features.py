import pandas as pd
import h5py
import numpy as np



def poser_base():
    df = pd.read_csv('../dataset/train_y.csv')
    index_labels = {i: df.id[df.sleep_stage == i].tolist() for i in range(5)}
    taille_min_sac = len(index_labels[1])
    sample = [np.random.choice(index_labels[i], taille_min_sac) for i in range(5)]
    all_samples = np.array([i for j in sample for i in j])
    np.random.shuffle(all_samples)
    train_samples = all_samples[:int(0.8 * len(all_samples))]
    val_samples = all_samples[int(0.8 * len(all_samples)):]
    return(train_samples, val_samples)

def extract_data(num):
    train_file = h5py.File('../dataset/train.h5', 'r')
    items = list(train_file.keys())
    TF = {}
    for index_item, item in enumerate(items):
        print(train_file[item][:].shape[1])
        if train_file[item][:].shape[1] == num:
            TF[item] = train_file[item][:]
    train_file.close()
    return TF

def finishing(train_samples, val_samples, TF, dimf, taille):
    X_train = {i: 0 for i in train_samples}
    for num in X_train.keys():
        all_data = []
        for index_item, item in enumerate(TF.keys()):
            all_data.append(list(TF[item][num]))
        X_train[num] = all_data
    X_val = {i: 0 for i in val_samples}
    for num in X_val.keys():
        all_data = []
        for index_item, item in enumerate(TF.keys()):
            all_data.append(list(TF[item][num]))
        X_val[num] = all_data
    Y_df = pd.read_csv('../dataset/train_y.csv')

    del TF

    df2 = pd.DataFrame(X_train)
    X_train_f = np.zeros((len(df2.keys()), dimf, taille))
    Y_train_f = np.zeros((len(df2.keys())))
    for j in range(len(df2.keys())):
        for k in range(7):
            st = df2[df2.keys()[j]][k]
            X_train_f[j][k][:] = st
        Y_train_f[j] = Y_df.sleep_stage[int(df2.keys()[j])]
    df3 = pd.DataFrame(X_val)
    X_val_f = np.zeros((len(df3.keys()), dimf, taille))
    Y_val_f = np.zeros((len(df3.keys())))
    for j in range(len(df3.keys())):
        for k in range(7):
            st = df3[df3.keys()[j]][k]
            X_val_f[j][k][:] = st
        Y_val_f[j] = Y_df.sleep_stage[int(df3.keys()[j])]

    return(df2, df3, X_train_f, X_val_f, Y_train_f, Y_val_f)

train_samples, val_samples = poser_base()
TF1500 = extract_data(1500)
(X_train_df_1500, X_val_df_1500, X_train_f1500, X_val_f1500, Y_train_f, Y_val_f) = finishing(train_samples, val_samples, TF1500, 7, 1500)
TF300 = extract_data(300)
(X_train_df_300, X_val_df_300, X_train_f300, X_val_f300, _, _) = finishing(train_samples, val_samples, TF300, 4, 300)



X_train_df_1500.to_csv("../dataset/X_features_train_equilibrate_1500.csv")
X_val_df_1500.to_csv("../dataset/X_features_val_equilibrate_1500.csv")
X_train_df_300.to_csv("../dataset/X_features_train_equilibrate_300.csv")
X_val_df_300.to_csv("../dataset/X_features_val_equilibrate_300.csv")
np.save("../dataset/X_features_train_equilibrate_300.npy", X_train_f300)
np.save("../dataset/X_features_train_equilibrate_1500.npy", X_train_f1500)
np.save("../dataset/Y_labels_train_equilibrate.npy", Y_train_f)
np.save("../dataset/X_features_val_equilibrate_300.npy", X_val_f300)
np.save("../dataset/X_features_val_equilibrate_1500.npy", X_val_f1500)
np.save("../dataset/Y_labels_val_equilibrate.npy", Y_val_f)
