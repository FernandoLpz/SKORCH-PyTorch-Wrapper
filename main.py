from torch import optim

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from skorch.callbacks import EpochScoring
from skorch import NeuralNetClassifier

from src import load_data
from src import NeuralNet

class Run:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def simple_training(self):
		# Trains the Neural Network with fixed hyperparameters

		# The Neural Net is initialized with fixed hyperparameters
		nn = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop)
		# Training
		nn.fit(self.x, self.y)
		pass

	def simple_pipeline_training(self):
		# Trains the Neural Network within a scikit-learn pipeline
		# The pipeline is composed by scaling features and NN training
		# The hyperparameters are fixed values

		# The Neural Net is initialized with fixed hyperparameters
		nn = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop)
		# The pipeline instatiated, it wraps scaling and training phase
		pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])
		# Pipeline execution
		pipeline.fit(self.x, self.y)

		pass

	def simple_pipeline_training_with_callbacks(self):
		# Trains the Neural Network within a scikit-learn pipeline
		# The pipeline is composed by scaling features and NN training
		# A callback is added in order to calculate the "balanced accuracy" and "accuracy" in the training phase

		# The EpochScoring from callbacks is initialized
		balanced_accuracy = EpochScoring(scoring='balanced_accuracy', lower_is_better=False)
		accuracy = EpochScoring(scoring='accuracy', lower_is_better=False)

		# The Neural Net is initialized with fixed hyperparameters
		nn = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop, callbacks=[balanced_accuracy, accuracy])
		# The pipeline instatiated, it wraps scaling and training phase
		pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])
		# Pipeline execution
		pipeline.fit(self.x, self.y)

		pass

	def grid_search_pipeline_training(self):
		# Through a grid search, the optimal hyperparameters are found
		# A pipeline is used in order to scale and train the neural net
		# The grid search module from scikit-learn wraps the pipeline

		# The Neural Net is instantiated, none hyperparameter is provided
		nn = NeuralNetClassifier(NeuralNet, verbose=0, train_split=False)
		# The pipeline is instantiated, it wraps scaling and training phase
		pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])

		# The parameters for the grid search are defined
		# It must be used the prefix "nn__" when setting hyperparamters for the training phase
		# It must be used the prefix "nn__module__" when setting hyperparameters for the Neural Net
		params = {
			'nn__max_epochs':[10, 20],
			'nn__lr': [0.1, 0.01],
			'nn__module__num_units': [5, 10],
			'nn__module__dropout': [0.1, 0.5],
			'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

		# The grid search module is instantiated
		gs = GridSearchCV(pipeline, params, refit=False, cv=3, scoring='balanced_accuracy', verbose=1)
		# Initialize grid search
		gs.fit(self.x, self.y)
		pass

if __name__ == "__main__":
	x, y = load_data()

	run = Run(x, y)

	# run.simple_training()
	# run.simple_pipeline_training()
	# run.simple_pipeline_training_with_callbacks()
	run.grid_search_pipeline_training()
