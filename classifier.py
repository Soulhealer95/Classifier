# Classifier that uses sklearn to work on a well defined dataset and compare classifier algorithms
import numpy as np
import sklearn.discriminant_analysis
import time
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.naive_bayes import GaussianNB as nb
DFLT_ATTRS = 4
DFLT_DATA = 1000
TRAINING_SIZE = 35000
# 25% of Training Size
TEST_SIZE = 8750 
ATTRIBUTE_COUNT = 7


class Classifier:

        # Binary Classification - Get data upon init
        # Assume first column is class id (binary only for now)
        # @param        [test_data_csv]         comma-separated testing data
        # @param        [train_data_csv]        comma-separated training data
        # @param        [attribute_limit]       maximum attributes to process
        # @param        [data_limit]            maximum training dataset to process
        # @param        [test_limit]            maximum testing dataset to process
        def __init__(self, test_data_csv, train_data_csv, attribute_limit=0, data_limit=0, test_limit = 0):
                self.test_data_raw = np.genfromtxt(test_data_csv, delimiter=',')
                self.train_data_raw = np.genfromtxt(train_data_csv, delimiter=',')

                # set default attributes if not provided
                if attribute_limit <= 0:
                        attribute_limit = DFLT_ATTRS
                if data_limit <= 0:
                        data_limit = DFLT_DATA
                if test_limit <= 0:
                        test_limit = DFLT_DATA
                self.limits = [attribute_limit, data_limit, test_limit]

                # other class variables to use
                self.test_x = []
                self.test_expect = []
                self.test_y = {
                                                "lda":[],       # Linear Discriminant
                                                "near": [],     # Nearest Neighbor
                                                "bayes": []     # Naive Bayes
                              }
                self.train_x = []
                self.train_y = []

        # Convert data into results and attributes
        #
        # @param        [data]              numpy array with data
        # @param        [is_test]           which limits to use for data (test or train)
        # @return       [outx, outy]        numpy arrays with limited data and results
        def __convert_data(self, data, is_test):
                outx = []
                outy = []
                if is_test == True:
                    limit = self.limits[2]
                else:
                    limit = self.limits[1]
                # get results first.
                outy = np.array(data[:,0])

                # separate the results
                outx = np.delete(data,np.s_[0], axis=1)

                # truncate data size per limits 1->rows 0->cols
                outy = np.delete(outy,np.s_[limit:-1], axis=0)
                outx = np.delete(outx,np.s_[limit:-1], axis=0)

                outx = np.delete(outx,np.s_[self.limits[0]-1:-1], axis=1)

                return outx, outy

        # Classify data using every classifier available
        # expects class methods of form <algorithm>_train() for each algorithm
        # defined in self.test_y
        # @param        [batch_mode]    Bool to enable/disable printing results
        # @return       [output]        Error Results for each algorithm
        def classify(self, batch_mode):
                # convert data to standard form
                self.train_x, self.train_y = self.__convert_data(self.train_data_raw, False)
                self.test_x, self.test_expect = self.__convert_data(self.test_data_raw, True)
                output = {}
                start_t = 0
                end_t = 0

                # call each available classifier to train
                for alg_id in list(self.test_y.keys()):
                        train_func = getattr(self, alg_id + "_train")
                        predict_func = getattr(self, alg_id + "_predict")

                        # time training and testing
                        start_t = time.time()
                        train_func()
                        end_t = time.time()
                        if batch_mode == False:
                            print(f'training time for {alg_id} : {round(end_t - start_t, 4)}s')

                        start_t = time.time()
                        predict_func()
                        end_t = time.time()
                        if batch_mode == False:
                            print(f'test time for {alg_id} : {round(end_t - start_t, 4)}s')

                        # get results 
                        output[alg_id] = self.get_results(alg_id, batch_mode)
                return output

        # Compare prediction with expected results of classification
        # generates confusion matrix and print results if not in batch mode
        #
        # @param        [alg_id]        String id for algorithm from self.test_y
        # @param        [batch_mode]    Bool to enable/disable printing results
        # @return       [output]        Error Results for each algorithm
        def get_results(self, alg_id, batch_mode):
            error = 0
            confusion = []
            false_pos = 0
            false_neg = 0
            true_pos = 0
            true_neg = 0
            cnt = len(self.test_expect)
            for i in range(cnt):
                    orig = self.test_expect[i]
                    calc = self.test_y[alg_id][i]
                    if calc != orig:
                        if orig == 1:
                                false_neg += 1
                        else:
                                false_pos += 1
                        error += 1
                    else:
                        if orig == 1:
                                true_pos += 1
                        else:
                                true_neg += 1

            confusion = np.array([[false_pos, true_pos], [true_neg, false_neg]])
            confusion = np.asmatrix(confusion)
            if batch_mode == False:
                print(f'results for {alg_id}:\n\
                        error count: {error}\n\
                        error rate: {round((error/cnt*100),3)}%\n\
                        confusion matrix:\n {confusion}')
            return [error, cnt, confusion]

        # Implement each classifier algorithm to train and predict.
        # currently simply calls sklearn method.
        def lda_train(self):
                self.lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
                self.lda.fit(self.train_x, self.train_y)
                return

        def lda_predict(self):
                self.test_y["lda"] = self.lda.predict(self.test_x)
                return

        # Nearest Neighbor
        def near_train(self):
                self.near = kn(n_neighbors=3)
                self.near.fit(self.train_x, self.train_y)
                return

        def near_predict(self):
                self.test_y["near"] = self.near.predict(self.test_x)
                return

        # Naive Bayes
        def bayes_train(self):
                self.bayes = nb()
                self.bayes.fit(self.train_x, self.train_y)
                return

        def bayes_predict(self):
                self.test_y["bayes"] = self.bayes.predict(self.test_x)
                return

# Main

# Run Once
c = Classifier("dota2Test.csv", "dota2Train.csv", ATTRIBUTE_COUNT, TRAINING_SIZE, TEST_SIZE)
# Batch Mode off
c.classify(False)

# Feature Selection Output. 
# Runs in Batch Mode Tries to Use Sequential Forward Generation
def feature_select():
    output = []
    for i in range(10):
        c = Classifier("dota2Test.csv", "dota2Train.csv", i+1, TRAINING_SIZE, TEST_SIZE)
        output.append(c.classify(True))

    for elem in output:
        print(elem)

# Uncomment for additional results 
# feature_select()
