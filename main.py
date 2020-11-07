
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

# TODO: It must be changed the classes labels
# instead of having: [1,2,3] it should be: [0,1,2]
class ClassifierModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu, dropout=0.5):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.dense0 = nn.Linear(13, num_units)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 4)

    def forward(self, X, **kwargs):
        X = F.relu(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        #X = F.softmax(self.output(X), dim=-1)
        X = F.softmax(self.output(X))

        return X

def load_data():
    data = pd.read_csv('data/wines.csv')
    x = data[[feature for feature in data.columns if feature != 'class']].values
    y = np.squeeze(data[['class']].values)

    x = x.astype(np.float32)
    y = y.astype(np.int64)

    return x,y 

if __name__ == "__main__":
    x, y = load_data()
    net = NeuralNetClassifier(ClassifierModule, max_epochs=20, lr=0.1, classes=[1,2,3])
    net.fit(x,y)
