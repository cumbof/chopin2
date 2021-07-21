#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Simone Truglia (s.truglia@students.uninettunouniversity.net)' )
__version__ = '0.01'
__date__ = 'Jul 21, 2021'

import sys, os, time, pickle, itertools, hashlib, math
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
                    default = 0,
                    help = "Number of retraining iterations" )
    p.add_argument( '--stop',
                    action = 'store_true',
                    default = False,
                    help = "Stop retraining if the error rate does not change" )
    p.add_argument( '--dataset', 
                    type = str,
                    help = "Path to the dataset file" )
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
                    type = str,
                    help = ( "Minimum and maximum number of features among those specified with the --features argument. " 
                             "It is equals to the number of features under --features if not specified. "
                             "Otherwise, it must be less or equals to the number of features under --features. "
                             "Warning: computationally intense! "
                             "Syntax: min:max" ) )
    p.add_argument( '--dump', 
                    action = 'store_true',
                    default = False,
                    help = "Build a summary and log files" )
    p.add_argument( '--cleanup', 
                    action = 'store_true',
                    default = False,
                    help = "Delete the classification model as soon as it produces the prediction accuracy" )
    p.add_argument( '--accuracy_threshold', 
                    type = float,
                    default = 60.0,
                    help = "Stop the execution if the best accuracy achieved during the previous group of runs is lower than this number" )
    p.add_argument( '--accuracy_uncertainty_perc', 
                    type = float,
                    default = 5.0,
                    help = ( "Take a run into account even if its accuracy is lower than the best accuracy achieved in the same group minus "
                             "its \"accuracy_uncertainty_perc\" percent" ) )
    p.add_argument( '--verbose',
                    action = 'store_true',
                    default = False,
                    help = "Print results in real time" )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'hdclass.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current hdclass.py version and exit" )
    # Apache Spark
    p.add_argument( '--spark',
                    action = 'store_true',
                    default = False,
                    help = "Build the classification model in a Apache Spark distributed environment" )
    p.add_argument( '--slices', 
                    type = int,
                    help = ( "Number of slices in case --spark argument is enabled. "
                             "This argument is ignored if --gpu is enabled" ) )
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
    return p.parse_args()

if __name__ == '__main__':
    # Load command line parameters
    args = read_params()
    fun.printlog( 
        'hdclass v{} ({})'.format( __version__, __date__ ),
        verbose=args.verbose
    )
    
    if args.pickle:
        fun.printlog( 
            'Loading features, trainData, trainLabels, testData, and testLabels from the pickle file',
            verbose=args.verbose
        )
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
            fun.printlog( 
                '\tFeatures not found! They will be initialised as incremental numbers',
                verbose=args.verbose
            )
            trainData, trainLabels, testData, testLabels = dataset
            features = [ str(f) for f in range( len( trainData[ 0 ] ) ) ]
    else:
        fun.printlog( 
            'Loading dataset\n\tDataset: {}\n\tTraining percentage: {}\n\tSeed: {}'.format( args.dataset,
                                                                                            args.training,
                                                                                            args.seed ),
            verbose=args.verbose
        )
        # Otherwise, split the dataset into training and test sets
        features, trainData, trainLabels, testData, testLabels = fun.buildDatasetPKL( args.dataset, 
                                                                                      separator=args.fieldsep,
                                                                                      training=args.training, 
                                                                                      seed=args.seed )
        # Dump pre-processed dataset to a pickle file
        pickledata = ( features, trainData, trainLabels, testData, testLabels )
        picklepath = os.path.join( os.path.dirname( args.dataset ), 
                                   '{}.pkl'.format( os.path.splitext( os.path.basename( args.dataset ) )[ 0 ] ) )
        fun.printlog( 
            'Dumping dataset to pickle file\n\t{}'.format( picklepath ),
            verbose=args.verbose
        )
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
                if len( use_features ) > len( recognised ):
                    unrecognised = list( set( use_features ).difference( set( recognised ) ) )
                    fun.printlog( 
                        'The following features cannot be recognised and will not be considered:',
                        data=unrecognised,
                        verbose=args.verbose
                    )
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
            summary_filepath = os.path.join( os.sep.join( picklepath.split( os.sep )[ :-1 ] ), 'summary.txt' )
            if not os.path.exists( summary_filepath ):
                with open( summary_filepath, 'a+' ) as summary:
                    fun.printlog( 
                        '# Run ID\tGroup Size\tDimensionality\tLevels\tRetraining\tBest Accuracy',
                        out=summary
                    )
        # Keep track of the feature selection result
        selected_features = [ ]
        # For each group size
        prev_group_size = np.Inf
        for group_size in reversed( range( min_group, max_group + 1 ) ):
            if group_size > 0 and group_size < prev_group_size:
                # Select the best set of features according to the last iteration
                # use the intersection of the features for the best runs
                # Otherwise, use the whole set of features
                best_features = use_features
                if group_size +1 in mapping:
                    best_accuracy = 0.0
                    best_features = [ ]
                    for run in mapping[ group_size +1 ]:
                        best_features = list( set(best_features).intersection(set(run[ "features" ])) ) if best_features else run[ "features" ]
                        if run["accuracy"] > best_accuracy:
                            best_accuracy = run["accuracy"]
                    if best_accuracy < args.accuracy_threshold:
                        # Stop the execution if the best accuracy achieved during the last group of runs is lower than "accuracy_threshold"
                        break
                    group_size = len( best_features ) -1
                    prev_group_size = group_size
                if group_size > 0:
                    # Create a directory for the current run
                    group_dir = os.path.join( os.sep.join( picklepath.split( os.sep )[ :-1 ] ), "HVs", "run_{}".format( group_size ) )
                    if not os.path.exists( group_dir ):
                        os.makedirs( group_dir )
                    # math.comb() has been introduced with python 3.8
                    #combinations = math.comb( len(best_features), group_size )
                    combinations = fun.combinations( len(best_features), group_size )
                    combinations_counter = 1
                    # Define a set of N features with N equals to "group_size"
                    for comb_features in itertools.combinations( best_features, group_size ):
                        # Build unique identifier for the current set of features
                        #features_hash = hash( tuple( comb_features ) ) % ( ( sys.maxsize + 1 ) * 2 )
                        features_hash = hashlib.md5(str(sorted(comb_features)).encode()).hexdigest()
                        # Create a log for the current run
                        run = None
                        copy_id = 0
                        run_filepath = os.path.join( group_dir, '{}_{}.log'.format( features_hash, copy_id ) )
                        # Do not overwrite old logs
                        while os.path.exists( run_filepath ):
                            copy_id += 1
                            run_filepath = os.path.join( group_dir, '{}_{}.log'.format( features_hash, copy_id ) )
                        if args.dump:
                            run = open( run_filepath, 'w+' )
                        # Print current run info and features
                        fun.printlog( 
                            'Run ID: {}\nCombination {}/{}\nGroup size: {}\nFeatures:'.format( '{}_{}'.format( features_hash, copy_id ),
                                                                                               int(combinations_counter), int(combinations), group_size ),
                            data=comb_features,
                            end_msg='Reshape training and test datasets',
                            verbose=args.verbose,
                            out=run
                        )
                        # Features positions
                        features_idx = [ ( feature in comb_features ) for feature in features ]
                        # Reshape trainData and testData if required
                        trainData_subset = trainData
                        testData_subset = testData
                        if len( comb_features ) < len( features ):
                            trainData_subset = [ [ value for index, value in enumerate( obs ) if features_idx[ index ] ] for obs in trainData ]
                            testData_subset = [ [ value for index, value in enumerate( obs ) if features_idx[ index ] ] for obs in testData ]
                        # Encodes the training data, testing data, and performs the initial training of the HD model
                        fun.printlog( 
                            'Build the HD model\n\t--dimensionality {}\n\t--levels {}'.format( args.dimensionality, args.levels ),
                            verbose=args.verbose,
                            out=run
                        )
                        t0model = time.time()
                        model = fun.buildHDModel( trainData_subset, trainLabels, testData_subset, testLabels, 
                                                args.dimensionality, args.levels, 
                                                os.path.splitext(
                                                    os.path.basename( picklepath )
                                                )[0],
                                                features_hash,
                                                workdir=group_dir,
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
                        fun.printlog( 
                            'Total elapsed time (model) {}s\nTest the HD model by retraining it {} times at most'.format( int( t1model - t0model ), args.retrain ),
                            verbose=args.verbose,
                            out=run
                        )
                        # Retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
                        t0acc = time.time()
                        accuracy = fun.trainNTimes( model.classHVs, 
                                                    model.trainHVs, model.trainLabels, 
                                                    model.testHVs, model.testLabels, 
                                                    args.retrain,
                                                    stop=args.stop,
                                                    spark=args.spark,
                                                    slices=args.slices,
                                                    master=args.master,
                                                    memory=args.memory,
                                                    dataset=os.path.splitext( os.path.basename( picklepath ) )[0],
                                                    verbose=args.verbose,
                                                    log=run
                                                  )
                        t1acc = time.time()
                        best = max( accuracy )
                        fun.printlog( 
                            'Total elapsed time (accuracy) {}s\nThe maximum accuracy is {} after {} retraining\n'.format( int( t1acc - t0acc ), best, len(accuracy) ),
                            verbose=args.verbose,
                            out=run
                        )
                        # Keep track of the best accuracy for the current group size
                        if group_size in mapping:
                            if best >= mapping[ group_size ][ 0 ][ "accuracy" ]:
                                threshold = best-(best*args.accuracy_uncertainty_perc)/100.0                                
                                mapping[ group_size ] = sorted( [ run for run in mapping[ group_size ] if run["accuracy"] >= threshold ], 
                                                                key=lambda run: run["accuracy"], 
                                                                reverse=True )
                                mapping[ group_size ].insert( 0,
                                    {
                                        "accuracy": best,
                                        "run": features_hash,
                                        "features": comb_features
                                    }
                                )
                        else:
                            mapping[ group_size ] = [
                                {
                                    "accuracy": best,
                                    "run": features_hash,
                                    "features": comb_features
                                }
                            ]
                        # Close log
                        if args.dump:
                            run.close()
                        # Cleanup
                        if args.cleanup:
                            fun.cleanup( group_dir, args.dimensionality, args.levels, features_hash, spark=args.spark )
                        combinations_counter += 1
                    # Keep track of the best results and close the summary
                    if args.dump:
                        if mapping[ group_size ][ 0 ][ "accuracy" ] >= args.accuracy_threshold:
                            with open( summary_filepath, 'a+' ) as summary:
                                selected_features = [ ]
                                for run in mapping[ group_size ]:
                                    selected_features.append(run["features"])
                                    fun.printlog( 
                                        '{}\t{}\t{}\t{}\t{}\t{}'.format( run[ "run" ], group_size, 
                                                                         args.dimensionality, args.levels, args.retrain, 
                                                                         run[ "accuracy" ] ),
                                        out=summary
                                    )
                                selected_features = list(set(list(itertools.chain(*selected_features))))
        
        fs_filepath = os.path.join( os.sep.join( picklepath.split( os.sep )[ :-1 ] ), 'selection.txt' )
        with open( fs_filepath, 'w+' ) as fs:
            for feature in selected_features:
                fun.printlog( feature, out=fs )
        fun.printlog( 'Selected features: {}'.format(fs_filepath), verbose=args.verbose )
