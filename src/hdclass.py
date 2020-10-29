#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Mar 27, 2020'

import os, time, pickle
import argparse as ap
import numpy as np
import functions as fun

def read_params():
    p = ap.ArgumentParser( description = ( "The hdclass.py script builds and tests a Hyperdimensional Computing (HDC) "
                                           "classification model on a input dataset." ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--dimensionality', 
                    type = int,
                    default = 10000,
                    help = "Dimensionality of the HDC model" )
    p.add_argument( '--levels', 
                    type = int,
                    help = "Number of level hypervectors" )
    p.add_argument( '--retrain', 
                    type = int,
                    default = 1,
                    help = "Number of retraining iterations" )
    p.add_argument( '--dataset', 
                    type = str,
                    help = "Path to the dataset pickle file" )
    p.add_argument( '--fieldsep', 
                    type = str,
                    default = ',',
                    help = "Field separator" )
    p.add_argument( '--training', 
                    type = int,
                    default = 80,
                    help = ( "Percentage of observations that will be used to train the model. "
                             "The remaining percentage will be used to test the classification model" ) )
    p.add_argument( '--seed', 
                    type = int,
                    default = 0,
                    help = ( "Seed for reproducing random sampling of the observations in the dataset "
                             "and build both the training and test set" ) )
    p.add_argument( '--pickle', 
                    type = str,
                    help = ( "Path to the pickle file. If specified, both '--dataset', '--fieldsep', "
                             "and '--training' parameters are not used" ) )
    # Apache Spark
    p.add_argument( '--spark',
                    action = 'store_true',
                    default = False,
                    help = "Build the classification model in a Apache Spark distributed environment" )
    p.add_argument( '--slices', 
                    type = int,
                    help = ( "Number of threads per block in case --gpu argument is enabled. "
                             "This argument is ignored if --spark is enabled" ) )
    p.add_argument( '--master', 
                    type = str,
                    help = "Master node address" )
    p.add_argument( '--memory', 
                    type = str,
                    default = "1024m",
                    help = "Executor memory" )
    # GPU parallelisation
    p.add_argument( '--gpu',
                    action = 'store_true',
                    default = False,
                    help = "Build the classification model on an NVidia powered GPU. "
                           "This argument is ignored if --spark is specified" )
    p.add_argument( '--tblock', 
                    type = int,
                    default = 32,
                    help = ( "Number of threads per block in case --gpu argument is enabled. "
                             "This argument is ignored if --spark is enabled" ) )
    # Standard multiprocessing
    p.add_argument( '--nproc', 
                    type = int,
                    default = 1,
                    help = ( "Number of parallel jobs for the creation of the HD model. "
                             "This argument is ignored if --spark is enabled" ) )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'hdclass.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current hdclass.py version and exit" )
    return p.parse_args()

if __name__ == '__main__':
    print( 'hdclass v{} ({})'.format( __version__, __date__ ) )
    # Load command line parameters
    args = read_params()

    if args.pickle:
        # If the pickle file already exists
        picklepath = args.pickle
        # Load trainData, trainLabels, testData, testLabels
        with open( picklepath, 'rb' ) as picklefile:
            dataset = pickle.load( picklefile )
        if len( dataset ) > 4:
            # Define features, trainData, trainLabels, testData, and testLabels
            features, trainData, trainLabels, testData, testLabels = dataset
        else:
            # Enable retro-compatibility for datasets with no features
            feature = list( range( len( trainData[ 0 ] ) ) )
            trainData, trainLabels, testData, testLabels = dataset
    else:
        # Otherwise, split the dataset into training and test sets
        features, trainData, trainLabels, testData, testLabels = fun.buildDataset( args.dataset, 
                                                                                   separator=args.fieldsep,
                                                                                   training=args.training, 
                                                                                   seed=args.seed )
        # Dump pre-processed dataset to a pickle file
        pickledata = ( features, trainData, trainLabels, testData, testLabels )
        picklepath = os.path.join( os.path.dirname( args.dataset ), 
                                   '{}.pkl'.format( os.path.splitext( os.path.basename( args.dataset ) )[ 0 ] ) )
        with open( picklepath, 'wb' ) as picklefile:
            pickle.dump( pickledata, picklefile )
    
    # features:     List of features
    # trainData:    Matrix in which each row is a datapoint of the training set and each column is a feature
    # trainLabels:  List in which each index contains the label for the data in the same row index of the trainData matrix
    # testData:     Matrix in which each row is a datapoint of the testing set and each column is a feature
    # testLabels:   List in which each index contains the label for the data in the same row index of the testData matrix
    if features and trainData and trainLabels and testData and testLabels:
        # Encodes the training data, testing data, and performs the initial training of the HD model
        t0model = time.time()
        model = fun.buildHDModel( features, trainData, trainLabels, testData, testLabels, 
                                  args.dimensionality, args.levels, 
                                  os.path.splitext(
                                    os.path.basename( picklepath )
                                  )[0],
                                  workdir=os.sep.join( picklepath.split( os.sep )[ :-1 ] ),
                                  spark=args.spark,
                                  slices=args.slices,
                                  master=args.master,
                                  memory=args.memory,
                                  gpu=args.gpu,
                                  tblock=args.tblock,
                                  nproc=args.nproc
                                )
        t1model = time.time()
        print( 'Total elapsed time (model) {}s'.format( int( t1model - t0model ) ) )
        # Retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
        t0acc = time.time()
        accuracy = fun.trainNTimes( model.classHVs, 
                                    model.trainHVs, model.trainLabels, 
                                    model.testHVs, model.testLabels, 
                                    args.retrain,
                                    spark=args.spark,
                                    slices=args.slices,
                                    master=args.master,
                                    memory=args.memory,
                                    dataset=os.path.splitext(
                                                    os.path.basename( picklepath )
                                                )[0]
                                  )
        t1acc = time.time()
        print( 'Total elapsed time (accuracy) {}s'.format( int( t1acc - t0acc ) ) )
        # Prints the maximum accuracy achieved
        print( 'The maximum accuracy is: ' + str( max( accuracy ) ) )