#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Mar 27, 2020'

import os, time, pickle
import argparse as ap
import numpy as np
import functions as hd

def read_params():
    p = ap.ArgumentParser( description = ( "The hdclass.py script builds and tests a Hyperdimensional Computing (HDC) classification model on a input dataset." ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--dimensionality', 
                    type = int,
                    default = 10000,
                    help = "Dimensionality of the HDC model." )
    p.add_argument( '--levels', 
                    type = int,
                    help = "Number of level hypervectors." )
    p.add_argument( '--retrain', 
                    type = int,
                    default = 1,
                    help = "Number of retraining iterations." )
    p.add_argument( '--dataset', 
                    type = str,
                    help = "Path to the dataset pickle file." )
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
    
    #the dataset shuld be split into 4 pats, each of which are numpy arrays:
    #trainData: this is a matrix where each row is a datapoint of the training set and each column is a feature
    #trainLabels: this is a array where each index contains the label for the data in the same row index of the trainData matrix
    #trainData: this is a matrix where each row is a datapoint of the testing set and each column is a feature
    #trainLabels: this is a array where each index contains the label for the data in the same row index of the testData matrix
    if os.path.exists(args.dataset):
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        trainData, trainLabels, testData, testLabels = dataset
        #encodes the training data, testing data, and performs the initial training of the HD model
        t0model = time.time()
        model = hd.buildHDModel(trainData, trainLabels, testData, testLabels, 
                                            args.dimensionality, args.levels, 
                                            os.path.splitext(
                                                os.path.basename(args.dataset)
                                            )[0],
                                            os.sep.join( args.dataset.split( os.sep )[ :-1 ] )
                                        )
        t1model = time.time()
        print('total elapsed time (model) {}s'.format(int(t1model - t0model)))
        #retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
        t0acc = time.time()
        accuracy = hd.trainNTimes(model.classHVs, 
                                            model.trainHVs, model.trainLabels, 
                                            model.testHVs, model.testLabels, 
                                            args.retrain
                                          )
        t1acc = time.time()
        print('total elapsed time (accuracy) {}s'.format(int(t1acc - t0acc)))
        #prints the maximum accuracy achieved
        print('the maximum accuracy is: ' + str(max(accuracy)))