#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Simone Truglia (s.truglia@students.uninettunouniversity.net)' )
__version__ = '0.01'
__date__ = 'Jul 21, 2021'

import os, random, copy, pickle, shutil, warnings, math
import numpy as np
import multiprocessing as mp
from functools import partial
warnings.filterwarnings("ignore")

# Try to load Spark
try:
    from pyspark import SparkConf, SparkContext
except:
    print("WARNING: PySpark not found")
    pass

# Try to load Numba
try:
    from numba import cuda
except:
    print("WARNING: Numba not found")
    pass

class HDModel(object):
    #Initializes a HDModel object
    #Inputs:
    #   datasetName; name of the dataset
    #   features: list of features
    #   trainData: training data
    #   trainLabels: training labels
    #   testData: testing data
    #   testLabels: testing labels
    #   D: dimensionality
    #   totalLevel: number of level hypervectors
    #   workdir: working directory
    #   spark: use Spark
    #   gpu: use GPU
    #   nproc: number of parallel jobs 
    #Outputs:
    #   HDModel object
    def __init__(self, datasetName, hashId, trainData, trainLabels, testData, testLabels, D, totalLevel, workdir, 
                 spark=False, slices=None, master=None, memory=None, gpu=False, tblock=32, nproc=1, 
                 verbose=False, log=None):
        if len(trainData) != len(trainLabels):
            printlog( "Training data and training labels do not have the same size", verbose=verbose, out=log )
            return
        if len(testData) != len(testLabels):
            printlog( "Testing data and testing labels do not have the same size", verbose=verbose, out=log )
            return
        self.datasetName = datasetName
        self.hashId = hashId
        self.workdir = workdir
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.D = D
        self.totalLevel = totalLevel
        self.levelList = getlevelList(self.trainData, self.totalLevel)
        levels_bufferHVs = os.path.join( self.workdir, 'levels_bufferHVs_{}_{}_{}.pkl'.format( str(self.D), str(self.totalLevel), str(self.hashId) ) )
        if os.path.exists( levels_bufferHVs ):
            printlog( "Loading HV levels\n\t{}".format( levels_bufferHVs ), verbose=verbose, out=log )
            self.levelHVs = pickle.load( open( levels_bufferHVs, 'rb' ) )
        else:
            printlog( "Generating HV levels\n\t{}".format( levels_bufferHVs ), verbose=verbose, out=log )
            self.levelHVs = genLevelHVs(self.totalLevel, self.D, gpu=gpu, tblock=tblock, verbose=verbose, log=log)
            with open( levels_bufferHVs, 'wb' ) as f:
                pickle.dump(self.levelHVs, f)
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []
        # Spark params
        self.spark = spark
        self.slices = slices
        self.master = master
        self.memory = memory
        # GPU params
        self.gpu = gpu
        self.tblock = tblock
        # Standard multiprocessing params
        self.nproc = nproc

    #Encodes the training or testing data into hypervectors and saves them or
    #loads the encoded traing or testing data that was saved previously
    #Inputs: 
    #   mode: decided to use train data or test data
    #Outputs:
    #   none
    def buildBufferHVs(self, mode, verbose=False, log=None):
        if self.spark:
            # Build a Spark context or use the existing one
            # Spark context must be initialised here (see SPARK-5063)
            config = SparkConf().setAppName( self.datasetName ).setMaster( self.master )
            config = config.set( 'spark.executor.memory', self.memory )
            context = SparkContext.getOrCreate( config )

        if mode == "train":
            train_bufferHVs = os.path.join( self.workdir, 'train_bufferHVs_{}_{}_{}{}'.format( str(self.D), str(self.totalLevel), str(self.hashId),
                                                                                               '.pkl' if not self.spark else '' ) )
            if os.path.exists( train_bufferHVs ):
                printlog( "Loading Encoded Training Data\n\t{}".format( train_bufferHVs ), verbose=verbose, out=log )
                if self.spark:
                    # Spark Context is running
                    trainHVs = context.pickleFile( train_bufferHVs )
                else:
                    with open( train_bufferHVs, 'rb' ) as f:
                        self.trainHVs = pickle.load(f)
            else:
                printlog( "Encoding Training Data\n\t{}".format( train_bufferHVs ), verbose=verbose, out=log )
                if self.spark:
                    # Spark Context is running
                    trainHVs = context.parallelize( list( zip( self.trainLabels, self.trainData ) ), numSlices=self.slices )
                    trainHVs = trainHVs.map( lambda label_obs: ( label_obs[0], EncodeToHV( label_obs[1], self.D, self.levelHVs, self.levelList ) ) )
                    trainHVs.saveAsPickleFile( train_bufferHVs )
                else:
                    # Multiprocessing
                    trainHVs = { }
                    with mp.Pool( processes=self.nproc ) as pool:
                        EncodeToHVPartial = partial( EncodeToHV_wrapper, 
                                                     D=self.D, 
                                                     levelHVs=self.levelHVs, 
                                                     levelList=self.levelList )

                        chunks = [ self.trainData[ i: i+self.nproc ] for i in range( 0, len(self.trainData), self.nproc ) ]
                        for cid in range( 0, len( chunks ) ):
                            positions = list( range( cid*len(chunks[ 0 ]), (cid*len(chunks[ 0 ]))+len(chunks[ cid ]) ) )
                            results = pool.starmap( EncodeToHVPartial, zip( chunks[ cid ], positions ) )
                            for position, vector in results:
                                trainHVs[ position ] = vector
                    self.trainHVs = [ vector for _, vector in sorted( trainHVs.items(), key=lambda item: item[ 0 ] ) ]
                    # Sequential
                    #for index in range(len(self.trainData)):
                    #    self.trainHVs.append(EncodeToHV(np.array(self.trainData[index]), self.D, self.levelHVs, self.levelList))
                    with open( train_bufferHVs, 'wb' ) as f:
                        pickle.dump(self.trainHVs, f)
            if self.spark:
                # Spark Context is running
                self.trainHVs = trainHVs.map( lambda obs: obs[ 1 ] ).collect()
                self.trainLabels = trainHVs.map( lambda obs: obs[ 0 ] ).collect()
                self.classHVs = trainHVs.reduceByKey( lambda obs1HV, obs2HV: np.add( obs1HV, obs2HV ) ).collectAsMap()
            else:
                self.classHVs = oneHvPerClass(self.trainLabels, self.trainHVs, gpu=self.gpu, tblock=self.tblock)
        else:
            test_bufferHVs = os.path.join( self.workdir, 'test_bufferHVs_{}_{}_{}{}'.format( str(self.D), str(self.totalLevel), str(self.hashId),
                                                                                             '.pkl' if not self.spark else '' ) )
            if os.path.exists( test_bufferHVs ):
                printlog( "Loading Encoded Testing Data\n\t{}".format( test_bufferHVs ), verbose=verbose, out=log )
                if self.spark:
                    # Spark Context is running
                    testHVs = context.pickleFile( test_bufferHVs )
                    self.testHVs = testHVs.map( lambda obs: obs[ 1 ] ).collect()
                    self.testLabels = testHVs.map( lambda obs: obs[ 0 ] ).collect()
                else:
                    with open( test_bufferHVs, 'rb' ) as f:
                        self.testHVs = pickle.load(f)
            else:
                printlog( "Encoding Testing Data\n\t{}".format( test_bufferHVs ), verbose=verbose, out=log )
                if self.spark:
                    # Spark Context is running
                    testHVs = context.parallelize( list( zip( self.testLabels, self.testData ) ), numSlices=self.slices )
                    testHVs = testHVs.map( lambda label_obs: ( label_obs[0], EncodeToHV( label_obs[1], self.D, self.levelHVs, self.levelList ) ) )
                    testHVs.saveAsPickleFile( test_bufferHVs )
                    self.testHVs = testHVs.map( lambda obs: obs[ 1 ] ).collect()
                    self.testLabels = testHVs.map( lambda obs: obs[ 0 ] ).collect()
                else:
                    # Multiprocessing
                    testHVs = { }
                    with mp.Pool( processes=self.nproc ) as pool:
                        EncodeToHVPartial = partial( EncodeToHV_wrapper, 
                                                     D=self.D, 
                                                     levelHVs=self.levelHVs, 
                                                     levelList=self.levelList )
                    
                        chunks = [ self.testData[ i: i+self.nproc ] for i in range( 0, len(self.testData), self.nproc ) ]
                        for cid in range( 0, len( chunks ) ):
                            positions = list( range( cid*len(chunks[ 0 ]), (cid*len(chunks[ 0 ]))+len(chunks[ cid ]) ) )
                            results = pool.starmap( EncodeToHVPartial, zip( chunks[ cid ], positions ) )
                            for position, vector in results:
                                testHVs[ position ] = vector
                    self.testHVs = [ vector for _, vector in sorted( testHVs.items(), key=lambda item: item[ 0 ] ) ]
                    # Sequential
                    #for index in range(len(self.testData)):
                    #    self.testHVs.append(EncodeToHV(np.array(self.testData[index]), self.D, self.levelHVs, self.levelList))
                    with open( test_bufferHVs, 'wb' ) as f:
                        pickle.dump(self.testHVs, f)
        if self.spark:
            # Stop Spark context
            context.stop()

# Define the following methods if Numba is available
try:
    @cuda.jit
    def gpu_add(A, B):
        pos = cuda.grid(1)
        if pos < A.size:
            A[pos] += B[pos]

    @cuda.jit
    def gpu_base(A, B):
        pos = cuda.grid(1)
        if pos < A.size:
            posB = A[pos]
            B[posB] = B[posB] * -1
except:
    pass

#Performs the initial training of the HD model by adding up all the training
#hypervectors that belong to each class to create each class hypervector
#Inputs:
#   inputLabels: training labels
#   inputHVs: encoded training data
#   gpu: use cuda
#   tblock: threads per block
#Outputs:
#   classHVs: class hypervectors
def oneHvPerClass(inputLabels, inputHVs, gpu=False, tblock=32):
    #This creates a dict with no duplicates
    classHVs = dict()
    for i in range(len(inputLabels)):
        name = inputLabels[i]
        if (name in classHVs.keys()):
            if gpu:
                A = np.array(classHVs[name])
                blocksPerGrid = (A.size + (tblock - 1))
                gpu_add[blocksPerGrid, tblock](A, np.array(inputHVs[i]))
                classHVs[name] = A
            else:
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
        prec_level = levelList[keyIndex]
        keyIndex = int((upperIndex + lowerIndex)/2)
        if (levelList[keyIndex] <= value and levelList[keyIndex+1] > value):
            return keyIndex
        if (levelList[keyIndex] > value):
            upperIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
        else:
            lowerIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex)/2)
        if levelList[keyIndex] == prec_level:
            lowerIndex = lowerIndex + 1
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
#   gpu: use cuda
#   tblock: threads per block
#Outputs:
#   levelHVs: level hypervector dictionary
def genLevelHVs(totalLevel, D, gpu=False, tblock=32, verbose=False, log=None):
    levelHVs = dict()
    indexVector = range(D)
    nextLevel = int((D/2/totalLevel))
    change = int(D / 2)
    for level in range(totalLevel):
        if level == 0:
            base = np.full( D, -1 )
            toOne = np.random.permutation(indexVector)[:change]
        else:
            toOne = np.random.permutation(indexVector)[:nextLevel]
        if gpu and toOne.size != 0:
            blocksPerGrid = (toOne.size + (tblock - 1))
            gpu_base[blocksPerGrid, tblock](toOne, base)
        else:
            for index in toOne:
                base[index] = base[index] * -1
        levelHVs[level] = copy.deepcopy(base)
    return levelHVs

def EncodeToHV_wrapper(inputBuffer, position, D=10000, levelHVs={}, levelList=[]):
    return position, EncodeToHV(inputBuffer, D, levelHVs, levelList)

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
def checkVector(classHVs, inputHV, labelHV):
    guess = list(classHVs.keys())[0]
    maximum = np.NINF
    count = {}
    for key in classHVs.keys():
        count[key] = inner_product(classHVs[key], inputHV)
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    if (labelHV == guess):
        return (1, guess)
    return (0, guess, labelHV, inputHV)

#Iterates through the training set once to retrain the model
#Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded train data
#   testLabels: training labels
#   spark: use Spark
#   dataset: dataset name
#Outputs:
#   retClassHVs: retrained class hypervectors
#   error: retraining error rate
def trainOneTime(classHVs, trainHVs, trainLabels, dataset="", verbose=False, log=None):
    retClassHVs = copy.deepcopy(classHVs)
    wrong_num = 0
    for index in range(len(trainLabels)):
        guess = checkVector(retClassHVs, trainHVs[index], trainLabels[index])[1]
        if not (trainLabels[index] == guess):
            wrong_num += 1
            retClassHVs[guess] = retClassHVs[guess] - trainHVs[index]
            retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + trainHVs[index]
    error = wrong_num / len(trainLabels)
    printlog( "\tError: {}".format(error), verbose=verbose, out=log )
    return retClassHVs, error

#Tests the HD model on the testing set
#Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded test data
#   testLabels: testing labels
#Outputs:
#   accuracy: test accuracy
def test(classHVs, testHVs, testLabels, spark=False, slices=None, master=None, memory=None, dataset="", verbose=False, log=None):
    if spark:
        # Build a Spark context or use the existing one
        # Spark context must be initialised here (see SPARK-5063)
        config = SparkConf().setAppName( dataset ).setMaster( master )
        config = config.set( 'spark.executor.memory', memory )
        context = SparkContext.getOrCreate( config )
        occTestTuple = list( zip( testLabels, testHVs ) )
        testRDD = context.parallelize( occTestTuple, numSlices=slices )
        correct = sum( testRDD.map( lambda label_hv: checkVector( classHVs, label_hv[1], label_hv[0] )[0] ).collect() )
        context.stop() 
    else:
        correct = 0
        for index in range(len(testHVs)):
            correct += checkVector(classHVs, testHVs[index], testLabels[index])[ 0 ]
    accuracy = (correct / len(testLabels)) * 100
    printlog( "\tThe accuracy is: {}".format(accuracy), verbose=verbose, out=log )
    return accuracy

#Retrains the HD model n times and evaluates the accuracy of the model
#after each retraining iteration
#Inputs:
#   classHVs: class hypervectors
#   trainHVs: encoded training data
#   trainLabels: training labels
#   testHVs: encoded test data
#   testLabels: testing labels
#   retrain: max retrain iterations
#   spark: use Spark
#   dataset: dataset name
#Outputs:
#   accuracy: array containing the accuracies after each retraining iteration
def trainNTimes(classHVs, trainHVs, trainLabels, testHVs, testLabels, retrain, stop=False,
                spark=False, slices=None, master=None, memory=None, dataset="", verbose=False, log=None):
    accuracy = []
    currClassHV = copy.deepcopy(classHVs)
    accuracy.append( test(currClassHV, testHVs, testLabels, 
                          spark=spark, slices=slices, master=master, memory=memory,
                          dataset=dataset, verbose=verbose, log=log) )
    prev_error = np.Inf
    for i in range(retrain):
        printlog( "Retraining iteration: {}".format(i+1), verbose=verbose, out=log )
        currClassHV, error = trainOneTime(currClassHV, trainHVs, trainLabels, 
                                          dataset=dataset, verbose=verbose, log=log)
        accuracy.append( test(currClassHV, testHVs, testLabels, 
                              spark=spark, slices=slices, master=master, memory=memory, 
                              dataset=dataset, verbose=verbose, log=log) )
        if error == prev_error and stop:
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
#   workdir: working directory
#   spark: use Spark
#   gpu: use GPU
#   nproc: number of parallel jobs
#Outputs:
#   model: HDModel object containing the encoded data, labels, and class HVs
def buildHDModel(trainData, trainLabels, testData, testLables, D, nLevels, datasetName, hash_id, workdir='./', 
                 spark=False, slices=None, master=None, memory=None,
                 gpu=False, tblock=32, nproc=1, 
                 verbose=False, log=None):
    # Initialise HDModel
    model = HDModel( datasetName, hash_id, trainData, trainLabels, testData, testLables, D, nLevels, workdir, 
                     spark=spark, slices=slices, master=master, memory=memory, gpu=gpu, tblock=tblock, nproc=nproc, 
                     verbose=verbose, log=log )
    # Build train HD vectors
    model.buildBufferHVs("train", verbose=verbose, log=log)
    # Build test HD vectors
    model.buildBufferHVs("test", verbose=verbose, log=log)
    return model

# Last line which starts with '#' will be considered header
# Last column contains classes
# First column contains the IDs of the observations
# Header line contains the feature names
def buildDatasetPKL( filepath, separator=',', training=80, seed=0 ):
    # Set a seed for the random sampling of the dataset
    random.seed( seed )
    # List of features
    features = [ ]
    # List of classes for each observation
    classes = [ ]
    # Observations content
    content = [ ]
    with open( filepath ) as file:
        # Trim the first and last columns out (Observation ID and Class)
        features = [ f.strip() for f in file.readline().split( separator )[ 1: -1 ] ]
        for line in file:
            line = line.strip()
            if line:
                if not line.startswith( '#' ):
                    line_split = line.split( separator )
                    content.append( [ float( value ) for value in line_split[ 1: -1 ] ] )
                    classes.append( line_split[ -1 ] )
    trainData = [ ]
    trainLabels = [ ]
    testData = [ ]
    testLabels = [ ]
    for classid in list( set( classes ) ):
        training_amount = int( ( float( classes.count( classid ) ) * float( training ) ) / 100.0 )
        # Create the training set by random sampling
        indices = [ pos for pos, val in enumerate(classes) if val == classid ]
        training_indices = random.sample( indices, training_amount )
        trainData.extend( [ content[ idx ] for idx in training_indices ] )
        trainLabels.extend( [ classid ]*len( training_indices ) )
        testData.extend( [ content[ idx ] for idx in indices if idx not in training_indices ] )
        testLabels.extend( [ classid ]*( len( indices )-len( training_indices ) ) )
    return features, trainData, trainLabels, testData, testLabels

# Rebuild the original CSV dataset starting from the PKL file
def buildDatasetFLAT( trainData, trainLabels, testData, testLabels, features, outpath, sep=',' ):
    data = trainData + testData
    labels = trainLabels + testLabels
    for observation in range( len( data ) ):
        data[ observation ].append( labels[ observation ] )
    features.append( 'class' )
    data.insert( 0, features )
    with open( outpath, 'w+' ) as flatfile:
        for observation in range( len( data ) ):
            flatfile.write( '{}\n'.format( sep.join( [ str(value) for value in data[ observation ] ] ) ) )

def printlog( message, data=[], print_threshold=100, end_msg=None, verbose=False, out=None ):
    if verbose:
        print( message )
    if out != None:
        out.write( '{}\n'.format( message ) )
    if data:
        exceeded = False
        if len(data) > print_threshold:
            print('\tWARNING: Exceeding maximum number of lines for printing ({}). Omitting {} lines.'.format(print_threshold, len(data)))
            exceeded = True
        for line in data:
            if verbose and not exceeded:
                print( '\t{}'.format( line ) )
            if out != None:
                out.write( '\t{}\n'.format( line ) )
    if end_msg != None:
        if verbose:
            print( end_msg )
        if out != None:
            out.write( '{}\n'.format( end_msg ) )

def cleanup( group_dir, dimensionality, levels, features_hash, spark=False ):
    suffix = 'bufferHVs_{}_{}_{}'.format( str(dimensionality), str(levels), str(features_hash) )
    for prefix in [ 'levels', 'train', 'test' ]:
        datapath = os.path.join( group_dir, '{}_{}'.format( prefix, suffix ) )
        if prefix == 'levels' or not spark:
            datapath = '{}.pkl'.format( datapath )
        if os.path.exists( datapath ):
            if os.path.isfile( datapath ):
                os.unlink( datapath )
            else:
                shutil.rmtree( datapath, ignore_errors=True )

def combinations(n, k):
    n_fac = math.factorial(n)
    k_fac = math.factorial(k)
    n_minus_k_fac = math.factorial(n - k)
    return n_fac/(k_fac*n_minus_k_fac)
