from torch import optim

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from skorch.callbacks import EpochScoring
from skorch import NeuralNetClassifier

from src import load_data
from src import NeuralNet

def simple_trainingl(x, y):
    net = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop)
    net.fit(x, y)
    pass

def simple_pipeline_training(x, y):
    net = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop)
    pipe = Pipeline([('scale', StandardScaler()), ('net', net)])
    pipe.fit(x, y)
    pass

def simple_pipeline_training_with_callbacks(x, y):
    balanced_accuracy = EpochScoring(scoring='balanced_accuracy', lower_is_better=False)
    accuracy = EpochScoring(scoring='accuracy', lower_is_better=False)

    net = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop, callbacks=[balanced_accuracy, accuracy])
    pipe = Pipeline([('scale', StandardScaler()), ('net', net)])
    pipe.fit(x, y)
    pass

def grid_search_pipeline_training(x, y):
    params = {
        'net__lr': [0.1, 0.01],
        'net__module__num_units': [5, 10],
        'net__module__dropout': [0.1, 0.5],
        'net__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

    net = NeuralNetClassifier(ClassifierModule, max_epochs=20, lr=0.1, verbose=0, train_split=False)
    pipe = Pipeline([('scale', StandardScaler()), ('net', net)])

    gs = GridSearchCV(pipe, params, refit=False, cv=3, scoring='balanced_accuracy', verbose=2)
    gs.fit(x, y)
    pass

if __name__ == "__main__":
    
    x, y = load_data()

    # PIPELINE
    balanced_accuracy = EpochScoring(scoring='balanced_accuracy', lower_is_better=False)
    accuracy = EpochScoring(scoring='accuracy', lower_is_better=False)
    net = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.Adam, callbacks=[balanced_accuracy, accuracy])
    # net.fit(x, y)
    pipe = Pipeline([('scale', StandardScaler()), ('net', net)])
    pipe.fit(x, y)

    print(', '.join(net.prefixes_))



    # GRID SEARCH & PIPELINE
    # params = {
    #     'net__lr': [0.1, 0.01],
    #     'net__module__num_units': [5, 10],
    #     'net__module__dropout': [0.1, 0.5],
    #     'net__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

    # net = NeuralNetClassifier(ClassifierModule, max_epochs=20, lr=0.1, verbose=0, train_split=False)
    # pipe = Pipeline([('scale', StandardScaler()), ('net', net)])

    # gs = GridSearchCV(pipe, params, refit=False, cv=3, scoring='balanced_accuracy', verbose=2)
    # gs.fit(x, y)