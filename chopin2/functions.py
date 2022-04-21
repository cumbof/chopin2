#!/usr/bin/env python

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Simone Truglia (s.truglia@students.uninettunouniversity.net)' )
__version__ = '1.0.6'
__date__ = 'Apr 21, 2022'

import os, random, copy, pickle, shutil, warnings, math
import numpy as np
import multiprocessing as mp
from functools import partial
warnings.filterwarnings("ignore")

def load_optional_modules(pyspark=False, numba=False, verbose=False):
    """
    Take control of the optional modules

    :pyspark:bool:      Load PySpark
    :numba:bool:        Load Numba
    :verbose:bool:      Print messages
    """

    if pyspark:
        # Try to load PySpark
        try:
            from pyspark import SparkConf, SparkContext
        except:
            if verbose:
                print("WARNING: PySpark not found")
            pass

    if numba:
        # Try to load Numba
        try:
            from numba import cuda
        except:
            if verbose:
                print("WARNING: Numba not found")
            pass

class HDModel(object):
    def __init__(self, datasetName, hashId, trainData, trainLabels, testData, testLabels, D, totalLevel, workdir, levelsdir,
                 k_fold=0, spark=False, slices=None, master=None, memory=None, gpu=False, tblock=32, nproc=1, 
                 verbose=False, log=None, seed=0):
        """
        Define a HDModel object

        :datasetName:str:           Dataset name or ID
        :hashId:str:                Unique identifier of the specific run
        :trainData:list:            Training dataset
        :trainLabels:list:          Class labels of the training observations
        :testData:list:             Test dataset
        :testLabels:list:           Class labels of the test observations
        :D:int:                     Dimensionality of the hypervectors
        :totalLevel:int:            Number of HD levels
        :workdir:str:               Path to the working directory
        :levelsdir:str:             Path to the folder in which the HD levels are located
        :k_fold:int:                Slice n-th for cross validation
        :spark:bool:                Enable the construction of the HD vectors on a Apache Spark distributed environment
        :slices:int:                Number of slices for distributing vectors over the Apache Spark cluster
        :master:str:                Address to the Apache Spark master node
        :memory:str:                Maximium memory allocated by Apache Spark
        :gpu:bool:                  Enable the construction of the HD vectors on NVidia GPUs
        :tblock:int:                Number of blocks on NVidia GPUs
        :nproc:int:                 Number of processors for running the construction of the hypervectors in multiprocessing
        :verbose:bool:              Print messages
        :log:str:                   Path to the log file
        :seed:int:                  Seed for reproducing the same random hypervectors
        """

        if len(trainData) != len(trainLabels):
            printlog( "Training data and training labels do not have the same size", verbose=verbose, out=log )
            return
        if len(testData) != len(testLabels):
            printlog( "Testing data and testing labels do not have the same size", verbose=verbose, out=log )
            return
        self.datasetName = datasetName
        self.hashId = hashId
        self.k_fold = k_fold
        self.workdir = workdir
        self.levelsdir = levelsdir
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.D = D
        self.totalLevel = totalLevel
        self.levelList = getlevelList(self.trainData, self.totalLevel)
        levels_bufferHVs = os.path.join( self.levelsdir, 'levels_bufferHVs_{}_{}.pkl'.format( str(self.D), str(self.totalLevel) ) )
        if os.path.exists( levels_bufferHVs ):
            printlog( "Loading HV levels\n\t{}".format( levels_bufferHVs ), verbose=verbose, out=log )
            self.levelHVs = pickle.load( open( levels_bufferHVs, 'rb' ) )
        else:
            printlog( "Generating HV levels\n\t{}".format( levels_bufferHVs ), verbose=verbose, out=log )
            self.levelHVs = genLevelHVs(self.totalLevel, self.D, gpu=gpu, tblock=tblock, verbose=verbose, log=log, seed=seed)
            with open( levels_bufferHVs, 'wb' ) as f:
                pickle.dump(self.levelHVs, f)
        self.trainHVs = list()
        self.testHVs = list()
        self.classHVs = list()
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

    def buildBufferHVs(self, mode, verbose=False, log=None):
        """
        Save or load already existing encoded training and test datasets

        :mode:str:          Two modes allowed: "train" and "test"
        :verbose:bool:      Print messages
        :log:str:           Path to the log file
        """

        if self.spark:
            # Build a Spark context or use the existing one
            # Spark context must be initialised here (see SPARK-5063)
            config = SparkConf().setAppName( self.datasetName ).setMaster( self.master )
            config = config.set( 'spark.executor.memory', self.memory )
            context = SparkContext.getOrCreate( config )

        if mode == "train":
            train_bufferHVs = os.path.join( self.workdir, 'train_bufferHVs_{}_{}_{}_{}{}'.format(self.D, self.totalLevel, self.hashId, self.k_fold,
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
                    trainHVs = dict()
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
            test_bufferHVs = os.path.join( self.workdir, 'test_bufferHVs_{}_{}_{}_{}{}'.format(self.D, self.totalLevel, self.hashId, self.k_fold,
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
                    testHVs = dict()
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

def oneHvPerClass(inputLabels, inputHVs, gpu=False, tblock=32):
    """
    Perform the initial training of the HD model by adding all the training hypervectors
    that belong to each class to create class hypervectors

    It returns class hypervectors

    :inputLabels:list:      Class labels of the input data
    :inputHVs:list:         HD encoded input data 
    :gpu:bool:              Enable the execution on NVidia GPUs
    :tblock:int:            Number of blocks on NVidia GPUs
    """
    
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
    """
    Compute the inner product on two hypervectors

    :x:numpy.array:     Hypervector x
    :y:numpy.array:     Hypervector y
    """

    return np.dot(x,y)  / (np.linalg.norm(x) * np.linalg.norm(y) + 0.0)

def numToKey(value, levelList):
    """
    Find the level hypervector index for the corresponding feature value

    :value:float:       Feature value
    :levelList:list:    List of level hypervector ranges
    """

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

def getlevelList(buffers, totalLevel):
    """
    Split the feature value range into level hypervector ranges

    It returns the list of level hypervector ranges

    :buffers:list:      Data matrix
    :totalLevel:int:    Number of level hypervector ranges
    """

    minimum = buffers[0][0]
    maximum = buffers[0][0]
    levelList = list()
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

def genLevelHVs(totalLevel, D, gpu=False, tblock=32, verbose=False, log=None, seed=0):
    """
    Generate the level hypervector dictionary

    :totalLevel:int:    Number of level hypervector
    :D:int:             Dimensionality of the hypervectors
    :gpu:bool:          Enable the generation of the HD levels on NVidia GPUs
    :tblock:int:        Number of blocks on NVidia GPUs
    :verbose:bool:      Print messages
    :log:str:           Path to the log file
    :seed:int:          Seed for reproducing the same random hypervectors
    """

    levelHVs = dict()
    indexVector = range(D)
    nextLevel = int((D/2/totalLevel))
    change = int(D / 2)
    for level in range(totalLevel):
        if level == 0:
            base = np.full( D, -1 )
            toOne = np.random.RandomState(seed=seed).permutation(indexVector)[:change]
        else:
            toOne = np.random.RandomState(seed=seed).permutation(indexVector)[:nextLevel]
        if gpu and toOne.size != 0:
            blocksPerGrid = (toOne.size + (tblock - 1))
            gpu_base[blocksPerGrid, tblock](toOne, base)
        else:
            for index in toOne:
                base[index] = base[index] * -1
        levelHVs[level] = copy.deepcopy(base)
    return levelHVs

def EncodeToHV_wrapper(inputBuffer, position, D=10000, levelHVs=dict(), levelList=list()):
    """
    Wrapper for the EncodeToHV function for multiprocessing

    :inputBuffer:list:      Input data that must be encoded
    :position:int           Position of chunk data for multiprocessing
    :D:int:                 Dimensionality of the hypervectors
    :levelHVs:dict:         Level hypervectors
    :levelList:list:        List of levels
    """

    return position, EncodeToHV(inputBuffer, D, levelHVs, levelList)

def EncodeToHV(inputBuffer, D, levelHVs, levelList):
    """
    Encode a single datapoint into a hypervector

    :inputBuffer:list:      Input data that must be encoded
    :D:int:                 Dimensionality of the hypervectors
    :levelHVs:dict:         Level hypervectors
    :levelList:list:        List of levels
    """

    sumHV = np.zeros(D, dtype = np.int)
    for keyVal in range(len(inputBuffer)):
        key = numToKey(inputBuffer[keyVal], levelList)
        levelHV = levelHVs[key] 
        sumHV = sumHV + np.roll(levelHV, keyVal)
    return sumHV
                    
def checkVector(classHVs, inputHV, labelHV):
    """
    Try to guess the class of the input vector based on the HD model

    :classHVs:dict:         Class hypervectors
    :inputHV:numpy.array:   Input hypervector
    :labelHV:str:           Label of the input hypervector
    """

    guess = list(classHVs.keys())[0]
    maximum = np.NINF
    count = dict()
    for key in classHVs.keys():
        count[key] = inner_product(classHVs[key], inputHV)
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    if (labelHV == guess):
        return (1, guess)
    return (0, guess, labelHV, inputHV)

def trainOneTime(classHVs, trainHVs, trainLabels, verbose=False, log=None):
    """
    Iterate over the training set once to retrain the HD model in order to minimize the error rate

    :classHVs:dict:         Class hypervectors
    :trainHVs:list:         Train hypervectors
    :trainLabels:list:      Label of the train hypervectors
    :verbose:bool:          Print messages
    :log:str:               Path to the log file
    """

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

def test(classHVs, testHVs, testLabels, spark=False, slices=None, master=None, memory=None, dataset="", verbose=False, log=None):
    """
    Test the HD model on the testing dataset and return the accuracy

    :classHVs:dict:         Class hypervectors
    :testHVs:list:          Test hypervectors
    :testLabels:list:       Label of the test hypervectors
    :spark:bool:            Enable the construction of the HD vectors on a Apache Spark distributed environment
    :slices:int:            Number of slices for distributing vectors over the Apache Spark cluster
    :master:str:            Address to the Apache Spark master node
    :memory:str:            Maximium memory allocated by Apache Spark
    :dataset:str:           Dataset name or ID
    :verbose:bool:          Print messages
    :log:str:               Path to the log file
    """

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

def trainNTimes(classHVs, trainHVs, trainLabels, testHVs, testLabels, retrain, stop=False,
                spark=False, slices=None, master=None, memory=None, dataset="", verbose=False, log=None):
    """
    Retrain the HD model N times and evaluate the accuracy of the model after each retraining iteration

    :classHVs:dict:         Class hypervectors
    :trainHVs:list:         Train hypervectors
    :trainLabels:list:      Label of the train hypervectors
    :testHVs:list:          Test hypervectors
    :testLabels:list:       Label of the test hypervectors
    :retrain:int:           Maximum number of retraining iterations
    :stop:bool:             Stop retraining if the error rate doesn't change with respect to the previous retraining iteration
    :spark:bool:            Enable the construction of the HD vectors on a Apache Spark distributed environment
    :slices:int:            Number of slices for distributing vectors over the Apache Spark cluster
    :master:str:            Address to the Apache Spark master node
    :memory:str:            Maximium memory allocated by Apache Spark
    :dataset:str:           Dataset name or ID
    :verbose:bool:          Print messages
    :log:str:               Path to the log file
    """

    accuracy = list()
    retraining = list()
    currClassHV = copy.deepcopy(classHVs)
    accuracy.append( test(currClassHV, testHVs, testLabels, 
                          spark=spark, slices=slices, master=master, memory=memory,
                          dataset=dataset, verbose=verbose, log=log) )
    retraining.append(0)
    prev_error = np.Inf
    for i in range(retrain):
        printlog( "Retraining iteration: {}".format(i+1), verbose=verbose, out=log )
        currClassHV, error = trainOneTime(currClassHV, trainHVs, trainLabels, 
                                          verbose=verbose, log=log)
        accuracy.append( test(currClassHV, testHVs, testLabels, 
                              spark=spark, slices=slices, master=master, memory=memory, 
                              dataset=dataset, verbose=verbose, log=log) )
        retraining.append(i+1)
        if error == prev_error and stop:
            break
        prev_error = error
    return accuracy, retraining

def buildHDModel(trainData, trainLabels, testData, testLables, D, nLevels, datasetName, hash_id, workdir='./', levelsdir='./', k_fold=0,
                 spark=False, slices=None, master=None, memory=None,
                 gpu=False, tblock=32, nproc=1, 
                 verbose=False, log=None, seed=0):
    """
    Create an HD model object, encode the training and test datasets, and perform the initial training of the HD model

    :trainData:list:        Train dataset
    :trainLabels:list:      Label of the train data
    :testData:list:         Test dataset
    :testLabels:list:       Label of the test data
    :D:int:                 Dimensionality of the hypervectors
    :nLevels:int:           Number of level hypervectors
    :datasetName:str:       Dataset name or ID
    :hash_id:str:           Unique identifier of the specific run
    :workdir:str:           Path to the working directory
    :levelsdir:str:         Path to the folder in which the HD levels are located
    :k_fold:int:            Slice n-th for cross validation
    :spark:bool:            Enable the construction of the HD vectors on a Apache Spark distributed environment
    :slices:int:            Number of slices for distributing vectors over the Apache Spark cluster
    :master:str:            Address to the Apache Spark master node
    :memory:str:            Maximium memory allocated by Apache Spark
    :gpu:bool:              Enable the generation of the HD levels on NVidia GPUs
    :tblock:int:            Number of blocks on NVidia GPUs
    :nproc:int:             Number of processors for running the construction of the hypervectors in multiprocessing
    :verbose:bool:          Print messages
    :log:str:               Path to the log file
    :seed:int:              Seed for reproducing the same random hypervectors
    """

    # Initialise HDModel
    model = HDModel( datasetName, hash_id, trainData, trainLabels, testData, testLables, D, nLevels, workdir, levelsdir, 
                     k_fold=k_fold, spark=spark, slices=slices, master=master, memory=memory, gpu=gpu, tblock=tblock, nproc=nproc, 
                     verbose=verbose, log=log, seed=seed )
    # Build train HD vectors
    model.buildBufferHVs("train", verbose=verbose, log=log)
    # Build test HD vectors
    model.buildBufferHVs("test", verbose=verbose, log=log)
    return model

def buildDatasetPKL( filepath, separator=',' ):
    """
    Build a PKL object with the training and test datasets

    :filepath:str:      Path to the dataset input file
    :separator:str:     Separator used to split fields in the input dataset
    :training:float:    Percentage of observations used to build the training (and implicitly the test) dataset
    :seed:int:          Seed for reproducing the same training and test datasets
    """

    # List of features
    features = list()
    # List of classes for each observation
    classes = list()
    # Observations content
    content = list()
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
    return features, content, classes

def percentage_split( data, classes, training=80.0, seed=0 ):
    """
    Build training and test sets with percentage split

    :data:list:         List with data
    :classes:list:      List of labels for each observation in data
    :training:float:    Percentage of observations used to build the training (and implicitly the test) dataset
    :seed:int:          Seed for reproducing the same training and test datasets
    """

    # Set a seed for the random sampling of the dataset
    random.seed( seed )

    trainData = list()
    trainLabels = list()
    testData = list()
    testLabels = list()
    for classid in list(set(classes)):
        training_amount = int((float(classes.count(classid))*float(training))/100.0)
        # Create the training set by random sampling
        indices = [pos for pos, val in enumerate(classes) if val == classid]
        training_indices = random.sample(indices, training_amount)
        trainData.extend([data[idx] for idx in training_indices])
        trainLabels.extend([classid]*len(training_indices))
        testData.extend([data[idx] for idx in indices if idx not in training_indices])
        testLabels.extend([classid]*(len(indices)-len(training_indices)))
    return trainData, trainLabels, testData, testLabels

def buildDatasetFLAT( data, labels, features, outpath, sep=',' ):
    """
    Build a dataset starting from the PKL object with training and test data

    :data:list:             Training dataset
    :labels:list:           Class labels of the training dataset
    :features:list:         Feature IDs
    :outpath:str:           Path to the output dataset file
    :sep:str:               Separator used to split fields in the output dataset
    """

    for observation in range( len( data ) ):
        data[ observation ].append( labels[ observation ] )
    features.append( 'class' )
    data.insert( 0, features )
    with open( outpath, 'w+' ) as flatfile:
        for observation in range( len( data ) ):
            flatfile.write( '{}\n'.format( sep.join( [ str(value) for value in data[ observation ] ] ) ) )

def printlog( message, data=list(), print_threshold=100, end_msg=None, verbose=False, out=None ):
    """
    Send messages to the stdout and/or to a file

    :message:str:               Custom message
    :data:list:                 List of data to be dumped to stdout/file
    :print_threshold:int:       Do not print data if its length is greater than this threshold
    :end_msg:str:               Print this message at the end
    :verbose:bool:              Print messages on the stdout
    :out:str:                   Print messages on file
    """

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

def cleanup( group_dir, levels_dir, dimensionality, levels, features_hash, k_folds=1, spark=False, skip_levels=False ):
    """
    Remove the HD model

    :group_dir:str:             Path to the run folder
    :levels_dir:str:            Path to the folder with the HD levels
    :dimensionality:int:        Dimensionality of the hypervectors
    :levels:int:                Number of HD levels
    :features_hash:str:         Unique identifier of the specific run
    :k_fold:int:                Slice n-th for cross validation
    :spark:bool:                Did you use Apache Spark?
    :skip_levels:bool:          Do not remove the HD levels
    """
    
    for k in range(k_folds):
        suffix = 'bufferHVs_{}_{}_{}_{}'.format(dimensionality, levels, features_hash, k)
        for prefix in [ 'train', 'test' ]:
            datapath = os.path.join( group_dir, '{}_{}'.format( prefix, suffix ) )
            if not spark:
                datapath = '{}.pkl'.format( datapath )
            if os.path.exists( datapath ):
                if os.path.isfile( datapath ):
                    os.unlink( datapath )
                else:
                    shutil.rmtree( datapath, ignore_errors=True )
    if not skip_levels:
        levels_datapath = os.path.join( levels_dir, 'levels_bufferHVs_{}_{}.pkl'.format( dimensionality, levels ) )
        if os.path.exists(levels_datapath):
            os.unlink(levels_datapath)
