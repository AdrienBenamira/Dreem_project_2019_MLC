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


def extract_features(num,  X_300, X_1500, Y, index_position):
    items = list(train_file.keys())
    compteur_300=0
    compteur_1500=0
    for index_item, item in enumerate(items):
        if len(train_file[item][index_position])==300:
            X_300[num][compteur_300] = train_file[item][index_position]
            compteur_300+=1
        else:
            X_1500[num][compteur_1500] = train_file[item][index_position]
            compteur_1500+=1
    Y[num]=df.sleep_stage[num]
    return(X_300,X_1500,Y)

def extract_all_features(train_samples):
    X_300 = np.zeros((len(train_samples), 4, 300))
    X_1500 = np.zeros((len(train_samples), len(list(train_file.keys()))-4, 1500))
    Y = np.zeros(len(train_samples))
    for index_sample, val_sample in enumerate(train_samples):
        if index_sample%1000 == 0:
            print(index_sample)
        X_300, X_1500, Y = extract_features(index_sample,  X_300, X_1500, Y, val_sample)
    return(X_300, X_1500, Y)

X_300, X_1500, Y = extract_all_features(train_samples)
X_300_val, X_1500_val, Y_val = extract_all_features(val_samples)

np.save("../dataset/X_features_train_equilibrate_300", X_300)
np.save("../dataset/X_features_train_equilibrate_1500", X_1500)
np.save("../dataset/Y_labels_train_equilibrate", Y)
np.save("../dataset/X_features_val_equilibrate_300", X_300_val)
np.save("../dataset/X_features_val_equilibrate_1500", X_1500_val)
np.save("../dataset/Y_labels_val_equilibrate", Y_val)
