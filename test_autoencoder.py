#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tools.data import DreemDatasets
from models.autoencoder import AE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.utils.data
import torch.optim


# In[2]:


use_cuda = torch.cuda.is_available()
# use_cuda = False


# In[3]:


use_datasets = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']
seed = 1

batch_size = 64
lr = 0.1
momentum = 0.5


# In[4]:


train_set, val_set = DreemDatasets('dataset/train.h5', 'dataset/train_y.csv', 
                                   split_train_val=0.8, seed=seed, keep_datasets=use_datasets,
                                   verbose=False).get()

train_set.load_data("dataset/all_eegs/train") 

val_set.load_data("dataset/all_eegs/val")

train_set.close()  # Ne ferme que les fichiers h5. Si mis en mémoire, on a toujours accès aux données !
val_set.close()


# In[5]:


ae = AE(1500*7, 100*7)

ae = ae.cuda() if use_cuda else ae

train_loader = torch.utils.data.DataLoader(train_set.torch_dataset(), batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(val_set.torch_dataset(), batch_size=batch_size, num_workers=1)

optimizer = torch.optim.SGD(ae.parameters(), lr=lr, momentum=momentum)


# In[6]:


def correlation_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return -cost


# In[7]:


ae.train()
for epoch in range(5):
    print("Epoch", epoch+1)
    for batch_id, (data_50hz, _, _) in enumerate(train_loader):
        if use_cuda:
            data_50hz = data_50hz.cuda()
        
        data_50hz = data_50hz.to(dtype=torch.float)
        optimizer.zero_grad()
        data_50hz = data_50hz.view(-1, 7*1500)
        out, inter = ae(data_50hz) 
        out = out.to(dtype=torch.float)
        # criterion = correlation_loss
        criterion = torch.nn.MSELoss(reduction='elementwise_mean')
        loss = criterion(data_50hz, out)
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print(loss)


# In[9]:


ae.eval()

X, _, y = train_set[:]
X_val, _, y_val = val_set[:]

X = X.numpy().transpose((1, 0, 2))
X = X.reshape(-1, 7*1500)

X_val = X_val.numpy().transpose((1, 0, 2))
X_val = X_val.reshape(-1, 7*1500)

X = torch.tensor(X)
X_val = torch.tensor(X_val)

_, z = ae(X)
_, z_val = ae(X_val)


# In[10]:


clf = RandomForestClassifier(n_estimators=100, random_state=0)


# In[12]:


clf.fit(z.detach().numpy(), y)


# In[14]:


labels_pred = clf.predict(z_val.detach().numpy())
cm = confusion_matrix(y_val, labels_pred)
acc = accuracy_score(y_val, labels_pred)
f1 = f1_score(y_val, labels_pred, average='macro')

print(cm, acc, f1)


# In[ ]:




