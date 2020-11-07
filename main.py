
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier

class ClassifierModule(nn.Module):
    def __init__(self, num_units=10, dropout=0.1):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.linear_1 = nn.Linear(13, num_units)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_units, 10)
        self.linear_3 = nn.Linear(10, 4)

    def forward(self, x, **kwargs):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.softmax(x, dim=-1)

        return x

def load_data():
    # Load csv dataset
    data = pd.read_csv('data/wines.csv')

    # Shuffling data
    data = shuffle(data)
    
    # Fix class labels
    # Original class labels are [1, 2, 3], the ones must be changed as [0, 1, 2]
    data['class'] = data['class'].replace([1, 2, 3],[0, 1, 2])

    # Split x and y vectors
    x = data[[feature for feature in data.columns if feature != 'class']].values
    y = np.squeeze(data[['class']].values)

    # Fix datatypes
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    return x, y 

if __name__ == "__main__":
    x, y = load_data()
    net = NeuralNetClassifier(ClassifierModule, max_epochs=30, lr=0.01, batch_size=12, optimizer=optim.Adam)
    # net.fit(x, y)

    pipe = Pipeline([('scale', StandardScaler()), ('net', net)])
    pipe.fit(x, y)