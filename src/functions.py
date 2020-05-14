import os, random, copy, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")

baseVal = -1

class HDModel(object):
    #Initializes a HDModel object
    #Inputs:
    #   trainData: training data
    #   trainLabels: training labels
    #   testData: testing data
    #   testLabels: testing labels
    #   D: dimensionality
    #   totalLevel: number of level hypervectors
    #Outputs:
    #   HDModel object
    def __init__(self, trainData, trainLabels, testData, testLabels, D, totalLevel, workdir):
        if len(trainData) != len(trainLabels):
            print("Training data and training labels are not the same size")
            return
        if len(testData) != len(testLabels):
            print("Testing data and testing labels are not the same size")
            return
        self.workdir = workdir
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.D = D
        self.totalLevel = totalLevel
        self.levelList = getlevelList(self.trainData, self.totalLevel)
        self.levelHVs = genLevelHVs(self.totalLevel, self.D)
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []

    #Encodes the training or testing data into hypervectors and saves them or
    #loads the encoded traing or testing data that was saved previously
    #Inputs: 
    #   mode: decided to use train data or test data
    #   D: dimensionality
    #   dataset: name of the dataset
    #Outputs:
    #   none
    def buildBufferHVs(self, mode, D, dataset):
        if mode == "train":
            train_bufferHVs = os.path.join( self.workdir, 'train_bufferHVs_{}.pkl'.format( str(D) ) )
            if os.path.exists( train_bufferHVs ):
                print("Loading Encoded Training Data")
                with open( train_bufferHVs, 'rb' ) as f:
                    self.trainHVs = pickle.load(f)
            else:       
                print("Encoding Training Data")
                for index in range(len(self.trainData)):
                    self.trainHVs.append(EncodeToHV(np.array(self.trainData[index]), self.D, self.levelHVs, self.levelList))
                with open( train_bufferHVs, 'wb' ) as f:
                    pickle.dump(self.trainHVs, f)
            self.classHVs = oneHvPerClass(self.trainLabels, self.trainHVs, self.D)
        else:
            test_bufferHVs = os.path.join( self.workdir, 'test_bufferHVs_{}.pkl'.format( str(D) ) )
            if os.path.exists( test_bufferHVs ):
                print("Loading Encoded Testing Data")
                with open( test_bufferHVs, 'rb' ) as f:
                    self.testHVs = pickle.load(f)
            else:
                print("Encoding Testing Data")       
                for index in range(len(self.testData)):
                    self.testHVs.append(EncodeToHV(np.array(self.testData[index]), self.D, self.levelHVs, self.levelList))
                with open( test_bufferHVs, 'wb' ) as f:
                    pickle.dump(self.testHVs, f)

#Performs the initial training of the HD model by adding up all the training
#hypervectors that belong to each class to create each class hypervector
#Inputs:
#   inputLabels: training labels
#   inputHVs: encoded training data
#   D: dimensionality
#Outputs:
#   classHVs: class hypervectors
def oneHvPerClass(inputLabels, inputHVs, D):
    #This creates a dict with no duplicates
    classHVs = dict()
    for i in range(len(inputLabels)):
        name = inputLabels[i]
        if (name in classHVs.keys()):
            classHVs[name] = np.array(classHVs[name]) + np.array(inputHVs[i])
        else:
            classHVs[name] = np.array(inputHVs[i])
    return classHVs

def inner_product(x, y):
    return np.dot(x,y)  / (np.linalg.norm(x) * np.linalg.norm(y) + 0.0)

#Finds the level hypervector index for the corresponding feature value
#Inputs:
#   value: feature value
#   levelList: list of level hypervector ranges
#Outputs:
#   keyIndex: index of the level hypervector in levelHVs corresponding the the input value
def numToKey(value, levelList):
    if (value == levelList[-1]):
        return len(levelList)-2
    upperIndex = len(levelList) - 1
    lowerIndex = 0
    keyIndex = 0
    while (upperIndex > lowerIndex):
        keyIndex = int((upperIndex + lowerIndex)/2)
        if (levelList[keyIndex] <= value and levelList[keyIndex+1] > value):
            return keyIndex
        if (levelList[keyIndex] > value):
            upperIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
        else:
            lowerIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
    return keyIndex  

#Splits up the feature value range into level hypervector ranges
#Inputs:
#   buffers: data matrix
#   totalLevel: number of level hypervector ranges
#Outputs:
#   levelList: list of the level hypervector ranges
def getlevelList(buffers, totalLevel):
    minimum = buffers[0][0]
    maximum = buffers[0][0]
    levelList = []
    for buffer in buffers:
        localMin = min(buffer)
        localMax = max(buffer)
        if (localMin < minimum):
            minimum = localMin
        if (localMax > maximum):
            maximum = localMax
    length = maximum - minimum
    gap = length / totalLevel
    for lv in range(totalLevel):
        levelList.append(minimum + lv*gap)
    levelList.append(maximum)
    return levelList

#Generates the level hypervector dictionary
#Inputs:
#   totalLevel: number of level hypervectors
#   D: dimensionality
#Outputs:
#   levelHVs: level hypervector dictionary
def genLevelHVs(totalLevel, D):
    print('generating level HVs')
    levelHVs = dict()
    indexVector = range(D)
    nextLevel = int((D/2/totalLevel))
    change = int(D / 2)
    for level in range(totalLevel):
        name = level
        if(level == 0):
            base = np.full(D, baseVal)
            toOne = np.random.permutation(indexVector)[:change]
        else:
            toOne = np.random.permutation(indexVector)[:nextLevel]
        for index in toOne:
            base[index] = base[index] * -1
        levelHVs[name] = copy.deepcopy(base)
    return levelHVs   

#Encodes a single datapoint into a hypervector
#Inputs:
#   inputBuffer: data to encode
#   D: dimensionality
#   levelHVs: level hypervector dictionary
#   IDHVs: ID hypervector dictionary
#Outputs:
#   sumHV: encoded data
def EncodeToHV(inputBuffer, D, levelHVs, levelList):
    sumHV = np.zeros(D, dtype = np.int)
    for keyVal in range(len(inputBuffer)):
        key = numToKey(inputBuffer[keyVal], levelList)
        levelHV = levelHVs[key] 
        sumHV = sumHV + np.roll(levelHV, keyVal)
    return sumHV
                    
# This function attempts to guess the class of the input vector based on the model given
#Inputs:
#   classHVs: class hypervectors
#   inputHV: query hypervector
#Outputs:
#   guess: class that the model classifies the query hypervector as
def checkVector(classHVs, inputHV):
    guess = list(classHVs.keys())[0]
    maximum = np.NINF
    count = {}
    for key in classHVs.keys():
        count[key] = inner_product(classHVs[key], inputHV)
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    return guess

#Iterates through the training set once to retrain the model
#Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded train data
#   testLabels: training labels
#Outputs:
#   retClassHVs: retrained class hypervectors
#   error: retraining error rate
def trainOneTime(classHVs, trainHVs, trainLabels):
    retClassHVs = copy.deepcopy(classHVs)
    wrong_num = 0
    for index in range(len(trainLabels)):
        guess = checkVector(retClassHVs, trainHVs[index])
        if not (trainLabels[index] == guess):
            wrong_num += 1
            retClassHVs[guess] = retClassHVs[guess] - trainHVs[index]
            retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + trainHVs[index]
    error = (wrong_num+0.0) / len(trainLabels)
    print('Error: {}'.format(error))
    return retClassHVs, error

#Tests the HD model on the testing set
#Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded test data
#   testLabels: testing labels
#Outputs:
#   accuracy: test accuracy
def test (classHVs, testHVs, testLabels):
    correct = 0
    for index in range(len(testHVs)):
        guess = checkVector(classHVs, testHVs[index])
        if (testLabels[index] == guess):
            correct += 1
    accuracy = (correct / len(testLabels)) * 100
    print('the accuracy is: {}'.format(accuracy))
    return (accuracy)

#Retrains the HD model n times and evaluates the accuracy of the model
#after each retraining iteration
#Inputs:
#   classHVs: class hypervectors
#   trainHVs: encoded training data
#   trainLabels: training labels
#   testHVs: encoded test data
#   testLabels: testing labels
#Outputs:
#   accuracy: array containing the accuracies after each retraining iteration
def trainNTimes (classHVs, trainHVs, trainLabels, testHVs, testLabels, n):
    accuracy = []
    currClassHV = copy.deepcopy(classHVs)
    accuracy.append(test(currClassHV, testHVs, testLabels))
    prev_error = np.Inf
    for i in range(n):
        print('iteration: {}'.format(i))
        currClassHV, error = trainOneTime(currClassHV, trainHVs, trainLabels)
        accuracy.append(test(currClassHV, testHVs, testLabels))
        if error == prev_error:
            break
        prev_error = error
    return accuracy

#Creates an HD model object, encodes the training and testing data, and
#performs the initial training of the HD model
#Inputs:
#   trainData: training set
#   trainLabes: training labels
#   testData: testing set
#   testLabels: testing labels
#   D: dimensionality
#   nLevels: number of level hypervectors
#   datasetName: name of the dataset
#Outputs:
#   model: HDModel object containing the encoded data, labels, and class HVs
def buildHDModel(trainData, trainLabels, testData, testLables, D, nLevels, datasetName, workdir='./'):
    model = HDModel(trainData, trainLabels, testData, testLables, D, nLevels, workdir)
    model.buildBufferHVs("train", D, datasetName)
    model.buildBufferHVs("test", D, datasetName)
    return model

