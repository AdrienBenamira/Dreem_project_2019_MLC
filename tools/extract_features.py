import pandas as pd
import h5py
import numpy as np


df = pd.read_csv('../dataset/train_y.csv')
train_file = h5py.File('../dataset/train.h5', 'r')

index_labels={i : df.index[df.sleep_stage==i].tolist() for i in range(5)}
taille_min_sac = len(index_labels[1])
sample = [np.random.choice(index_labels[i], taille_min_sac) for i in range(5)]
all_samples = np.array([i for j in sample for i in j])
np.random.shuffle(all_samples)
train_samples = all_samples[:int(0.8*len(all_samples))]
val_samples = all_samples[int(0.8*len(all_samples)):]


def extract_features(train_samples):
    X_300 = np.zeros((4, len(train_samples), 300))
    X_1500 = np.zeros((len(list(train_file.keys())) - 4, len(train_samples), 1500))
    Y = np.zeros(len(train_samples))
    items = list(train_file.keys())
    compteur_300 = 0
    compteur_1500 = 0
    for index_item, item in enumerate(items):
        train_fi = train_file[item][:]
        print(train_fi.shape)
        if train_fi.shape[1] == 300:
            X_300[compteur_300] = np.take(train_fi, train_samples, axis=0)
            compteur_300 += 1
        else:
            X_1500[compteur_1500] = np.take(train_fi, train_samples, axis=0)
            compteur_1500 += 1
    for index_sample, val_sample in enumerate(train_samples):
        Y[index_sample] = df.sleep_stage[val_sample]
    return(X_300, X_1500, Y)

X_300, X_1500, Y = extract_features(train_samples)
X_300_val, X_1500_val, Y_val = extract_features(val_samples)

X_300.shape
X_300s = X_300.reshape(5412,4,300)
print(X_300.shape)

print(X_1500.shape)
X_1500s = X_1500.reshape(5412,7,1500)
print(X_1500.shape)

print(X_300_val.shape)
X_300_vals = X_300_val.reshape(1353,4,300)
print(X_300_val.shape)

X_1500_val.shape
X_1500_vals = X_1500_val.reshape(1353,7,1500)
print(X_1500_val.shape)

np.save("../dataset/X_features_train_equilibrate_300.npy", X_300s)
np.save("../dataset/X_features_train_equilibrate_1500.npy", X_1500s)
np.save("../dataset/Y_labels_train_equilibrate.npy", Y)
np.save("../dataset/X_features_val_equilibrate_300.npy", X_300_vals)
np.save("../dataset/X_features_val_equilibrate_1500.npy", X_1500_vals)
np.save("../dataset/Y_labels_val_equilibrate.npy", Y_val)

train_file.close()
