
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

import pandas as pd

class ClassifierModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu, dropout=0.5):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X

def load_data():
    data = pd.read_csv('data/wines.csv')
    x = data[[feature for feature in data.columns if feature != 'class']]
    y = data[['class']]
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")

    return x,y 

if __name__ == "__main__":
    x, y = load_data()
    net = NeuralNetClassifier(ClassifierModule, max_epochs=20, lr=0.1)

