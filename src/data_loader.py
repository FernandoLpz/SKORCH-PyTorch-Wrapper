import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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