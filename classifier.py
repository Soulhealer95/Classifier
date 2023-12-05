# Classifier class that uses sklearn to work on a well defined dataset and compare two classifier algorithms
import numpy as np
import sklearn.discriminant_analysis


class Classifier:

	# Binary Classification - Get data upon init
	# Assume first column is class id (binary only for now)
	# @param	[test_data_csv]		comma-separated testing data
	# @param	[train_data_csv]	comma-separated training data
	def __init__(self, test_data_csv, train_data_csv):
		# we'll limit attributes to 4 and data size to 1000
		self.limits = [4,1000]
		self.test_data_raw = np.genfromtxt(test_data_csv, delimiter=',')
		self.train_data_raw = np.genfromtxt(train_data_csv, delimiter=',')
		self.test_x = []
		self.test_expect = []
		self.test_y = {"lda":[]}
		self.train_x = []
		self.train_y = []

	# Convert data into results and attributes
	#
	# @param	[data]			numpy array with data
	# @return	[outx, outy]	numpy arrays with limited data and results
	def __convert_data(self, data):
		outx = []
		outy = []
		# get results first.
		outy = np.array(data[:,0])

		# separate the results
		outx = np.delete(data,np.s_[0], axis=1)

		# truncate data size per limits 1->rows 0->cols
		outy = np.delete(outy,np.s_[self.limits[1]:-1], axis=0)
		outx = np.delete(outx,np.s_[self.limits[1]:-1], axis=0)

		outx = np.delete(outx,np.s_[self.limits[0]-1:-1], axis=1)

		return outx, outy

	# Classify data using every classifier available
	# expects class methods of form <algorithm>_train() for each algorithm
	# defined in self.test_y
	def classify(self):
		# convert data to standard form
		self.train_x, self.train_y = self.__convert_data(self.train_data_raw)
		self.test_x, self.test_expect = self.__convert_data(self.test_data_raw)

		# call each available classifier to train
		for alg_id in list(self.test_y.keys()):
			train_func = getattr(self, alg_id + "_train")
			predict_func = getattr(self, alg_id + "_predict")
			train_func()
			predict_func()
			self.get_error(alg_id)
		return

	# Compare prediction with expected results of classification and print error
	# @param	[alg_id]	String id for algorithm from self.test_y
	def get_error(self, alg_id):
		error = 0
		cnt = len(self.test_expect)
		for i in range(cnt):
			if self.test_y[alg_id][i] != self.test_expect[i]:
					error += 1
		print(f'results for {alg_id}:\n\
				error count: {error}\n\
				error rate: {round((error/cnt*100),3)}%')
		return error

	# Implement each classifier algorithm to train and predict.
	# currently simply calls sklearn method.
	def lda_train(self):
		self.lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
		self.lda.fit(self.train_x, self.train_y)
		return

	def lda_predict(self):
		self.test_y["lda"] = self.lda.predict(self.test_x)
		return


# Main
c = Classifier("dota2Test.csv", "dota2Train.csv")
c.classify()
