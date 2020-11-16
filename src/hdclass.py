#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Mar 27, 2020'

import sys, os, time, pickle, itertools, hashlib
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
    p.add_argument( '--features', 
                    type = str,
                    help = "Path to a file with a single column containing the whole set or a subset of feature" )
    p.add_argument( '--group',
                    type = int,
                    help = ( "Minimum and maximum number of features among those specified with the --features argument. " 
                             "It is equals to the number of features under --features if not specified. "
                             "Otherwise, it must be less or equals to the number of features under --features. "
                             "Warning: computationally intense! "
                             "Syntax: min:max" ) )
    p.add_argument( '--dump', 
                    type = str,
                    help = "Path to the output file with classification results" )
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
                    help = ( "Build the classification model on an NVidia powered GPU. "
                             "This argument is ignored if --spark is specified" ) )
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
    p.add_argument( '--verbose',
                    action = 'store_true',
                    default = False,
                    help = "Print results in real time" )
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
            trainData, trainLabels, testData, testLabels = dataset
            feature = [ str(f) for f in range( len( trainData[ 0 ] ) ) ]
    else:
        # Otherwise, split the dataset into training and test sets
        features, trainData, trainLabels, testData, testLabels = fun.buildDatasetPKL( args.dataset, 
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
        # Define the set of features that must be considered for building the HD model
        use_features = [ ]
        if args.features:
            with open( args.features ) as features_list:
                for line in features_list:
                    line = line.strip()
                    if line:
                        use_features.append( line )
            recognised = list( set( use_features ).intersection( set( features ) ) )
            if args.verbose:
                if len( use_features ) > recognised:
                    print( 'The following features cannot be recognised and will not be considered: ' )
                    unrecognised = list( set( use_features ).difference( set( recognised ) ) )
                    for feature in unrecognised:
                        print( '\t{}'.format( feature ) )
        else:
            use_features = features
        # Define the minimum and maximum number of features per group
        if args.group:
            try:
                min_group, max_group = [ int(g) for g in args.group.split( ':' ) ]
                if min_group > max_group:
                    raise Exception( ( "Incorrect --group argument! "
                                       "Left end must be greater than or equals to the right end" ) )
            except:
                raise Exception( ( "Incorrect --group argument! "
                                   "Interval must contain numbers" ) )
        else:
            min_group = len( use_features )
            max_group = len( use_features )
        # Keep track of the best <group_size, accuracy>
        mapping = { }
        # Create the summary file
        if args.dump:
            summary = open( os.path.join( args.dump, 'summary.txt' ), 'w+' )
            summary.write( '# Group Size\tBest Accuracy\tRun ID\n' )
        # For each group size
        for group_size in range( min_group, max_group + 1 ):
            # Create a directory for the current run
            if args.dump:
                group_dir = os.path.join( args.dump, "batch_{}".format( group_size ) )
                if not os.path.exists( group_dir ):
                    os.mkdir( group_dir )
            # Define a set of N features with N equals to "group_size"
            for comb_features in itertools.combinations( use_features, group_size ):
                # Build unique identifier for the current set of features
                #features_hash = hash( tuple( comb_features ) ) % ( ( sys.maxsize + 1 ) * 2 )
                features_hash = hashlib.md5(str(sorted(comb_features)).encode()).hexdigest()
                # Create a log for the current run
                run = None
                if args.dump:
                    run = open( os.path.join( group_dir, '{}.log'.format( features_hash ) ), 'w+' )
                    run.write( 'Run ID: {}\n'.format( features_hash ) )
                    run.write( 'Group size: {}\n'.format( group_size ) )
                # Print current run info
                if args.verbose:
                    print( 'Run ID: {}'.format( features_hash ) )
                    print( 'Group size: {}'.format( group_size ) )
                # Print the current set of features
                if args.verbose:
                    print( 'Features:' )
                    for feature in comb_features:
                        print( '\t{}'.format( feature ) )
                    print( 'Reshape training and test datasets' )
                # Keep track of the features in log
                if args.dump:
                    run.write( 'Features:\n' )
                    for feature in comb_features:
                        run.write( '\t{}\n'.format( feature ) )
                # Features positions
                features_idx = [ ( feature in comb_features ) for feature in features ]
                # Reshape trainData and testData if required
                trainData_subset = trainData
                testData_subset = testData
                if len( comb_features ) < len( features ):
                    trainData_subset = [ [ value for index, value in enumerate( obs ) if features_idx[ index ] ] for obs in trainData ]
                    testData_subset = [ [ value for index, value in enumerate( obs ) if features_idx[ index ] ] for obs in testData ]
                # Encodes the training data, testing data, and performs the initial training of the HD model
                if args.verbose:
                    print( 'Build the HD model' )
                t0model = time.time()
                model = fun.buildHDModel( trainData_subset, trainLabels, testData_subset, testLabels, 
                                          args.dimensionality, args.levels, 
                                          os.path.splitext(
                                            os.path.basename( picklepath )
                                          )[0],
                                          features_hash,
                                          workdir=os.sep.join( picklepath.split( os.sep )[ :-1 ] ),
                                          spark=args.spark,
                                          slices=args.slices,
                                          master=args.master,
                                          memory=args.memory,
                                          gpu=args.gpu,
                                          tblock=args.tblock,
                                          nproc=args.nproc,
                                          verbose=args.verbose,
                                          log=run
                                        )
                t1model = time.time()
                if args.verbose:
                    print( 'Total elapsed time (model) {}s'.format( int( t1model - t0model ) ) )
                    print( 'Test the HD model by retraining it {} times'.format( args.retrain ) )
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
                                                    )[0],
                                            verbose=args.verbose,
                                            log=run
                                          )
                t1acc = time.time()
                best = max( accuracy )
                if args.verbose:
                    print( 'Total elapsed time (accuracy) {}s'.format( int( t1acc - t0acc ) ) )
                    # Prints the maximum accuracy achieved
                    print( 'The maximum accuracy is: ' + str( best ) )
                # Keep track of the best accuracy for the current group size
                if group_size in mapping:
                    if mapping[ group_size ][ "accuracy" ] < best:
                        mapping[ group_size ][ "accuracy" ] = best
                        mapping[ group_size ][ "run" ] = features_hash
                else:
                    mapping[ group_size ] = {
                        "accuracy": best,
                        "run": features_hash
                    }
                # Close log
                if args.dump:
                    run.close()
        # Keep track of the best results and close the summary
        if args.dump:
            for group_size in sorted(mapping.keys()):
                summary.write( '{}\t{}\t{}\n'.format( group_size, 
                                                      mapping[ group_size ][ "accuracy" ],
                                                      mapping[ group_size ][ "run" ] ) )
            # Close summary
            summary.close()