# Homework 2-1 Phoneme Classification

########## Download Data ##########
!pip install --upgrade gdown

# Main link
# !gdown --id '1N1eVIDe9hKM5uiNRGmifBlwSDGiVXPJe' --output data.zip
!gdown --id '1qzCRnywKh30mTbWUEjXuNT2isOCAPdO1' --output data.zip

!unzip -q data.zip
!ls data

########## Preparing Data ########## 

import numpy as np
print('Loading data ...')

data_root='./data/'
train = np.load(data_root + 'data.npy')
train_label = np.load(data_root + 'train_label.npy')
test = np.load(data_root + 'test.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

from collections import Counter
counter = Counter(train_label)
for k,v in counter.items():
	per = v / len(train_label) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

import pandas as pd
df = pd.read_csv("/content/distribution.csv")
df['count'] = df['count'] / 1000
df.head()

import matplotlib.pyplot as plt
df.plot(x="id", y="count")
plt.xlabel('class')
plt.ylabel('count(k times)')
plt.show()

def sample(data, stride):
  base = data.copy()
  for step in range(1, stride+1, 1):
    Rshift = np.roll(base,step,axis=0)
    data = np.concatenate((Rshift,data), axis=1)
  for step in range(-1, -1-1*stride, -1):
    Lshift = np.roll(base,step,axis=0)
    data = np.concatenate((data,Lshift), axis=1)

  data = np.reshape(data, (-1,2*stride+1,39))
  return data

def normalize(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  return (data - mean)/std

stride = 10
# (,429)split to(11,39)
train = np.reshape(train, (-1,11,39))
test = np.reshape(test, (-1,11,39))

# pick only the 5th MFCC which is corresponding to label
train = train[:,5,:]
test = test[:,5,:]

# include nearby MFCC (To extend the frame length)
train = sample(train, stride)
test = sample(test, stride)

# flatten to (-1,39*n)
train = np.reshape(train, (-1,(2*stride+1)*39))
test = np.reshape(test, (-1,(2*stride+1)*39))

# normalize data
# train = normalize(train)
# test = normalize(test)
     

import matplotlib.pyplot as plt
mfcc = np.reshape(train[0], ((2*stride+1),39))
fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(mfcc);


########## Create Dataset ########## 

import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

VAL_RATIO = 0
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

BATCH_SIZE = 1024
from torch.utils.data import DataLoader
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

########## Create Model ########## 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
        #1
        nn.Linear((2*stride+1)*39, 1024),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.ReLU(),
        #2
        nn.Linear(1024, 2048),
        nn.BatchNorm1d(2048),
        nn.Dropout(0.25),
        nn.ReLU(),
        #3
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.Dropout(0.25),
        nn.ReLU(),
        #4
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.ReLU(),
        #5
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.ReLU(),
        #6
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        nn.ReLU(),
        
        #7
        nn.Linear(128, 39)
        )
        

    def forward(self, x):
        x = self.net(x)
        x = F.log_softmax(x, dim=1)
        return x

########## Training ########## 

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 500               # number of training epoch
learning_rate = 1e-4       # learning rate
l2 = 0
# the path where checkpoint saved
model_path = './modelFC500.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

# start training
best_acc = 0.0
Total_loss = []
Total_acc = []
for epoch in range(num_epoch):
    #switch optimizer
    if epoch == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif epoch == 35:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()
        

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        Total_acc.append(train_acc/len(train_set))
        Total_loss.append(train_loss/len(train_loader))
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
