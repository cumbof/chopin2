import HDFunctions
import numpy as np
import pickle
import time

#dimensionality of the HDC model
D = 10000
#number of level hypervectors
nLevels = 100
#number of retraining iterations
n = 20
#loading the isolet dataset (load your desired dataset here)
#the dataset shuld be split into 4 pats, each of which are numpy arrays:
#trainData: this is a matrix where each row is a datapoint of the training set and each column is a feature
#trainLabels: this is a array where each index contains the label for the data in the same row index of the trainData matrix
#trainData: this is a matrix where each row is a datapoint of the testing set and each column is a feature
#trainLabels: this is a array where each index contains the label for the data in the same row index of the testData matrix
with open('./../dataset/isolet/isolet.pkl', 'rb') as f:
    isolet = pickle.load(f)
trainData, trainLabels, testData, testLabels = isolet
#encodes the training data, testing data, and performs the initial training of the HD model
t0model = time.time()
model = HDFunctions.buildHDModel(trainData, trainLabels, testData, testLabels, D, nLevels, 'isolet')
t1model = time.time()
print('total elapsed time (model) {}s'.format(int(t1model - t0model)))
#retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
t0acc = time.time()
accuracy = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.testHVs, model.testLabels, n)
t1acc = time.time()
print('total elapsed time (accuracy) {}s'.format(int(t1acc - t0acc)))
#prints the maximum accuracy achieved
print('the maximum accuracy is: ' + str(max(accuracy)))