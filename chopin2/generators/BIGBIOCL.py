import os, pickle, random

features = [ ]
trainData = [ ]
trainLabels = [ ]
testData = [ ]
testLabels = [ ]

# We just need the training percentage
# the test percentage is 100.0 - training_perc
training_perc = 80.0

basepath = "/Users/fabio/GitHub/HD-Classifier/datasets/BIGBIOCL/thca_dnameth/"
inputname = "thca_dnameth_NOCONTROL"
inputfile = os.path.join( basepath, "{}.csv".format(inputname) )

classes = { }
with open( inputfile ) as dataset:
    # skip header
    features = dataset.readline().strip().split( ',' )[ 1: -1 ]
    # read dataset
    for line in dataset:
        line = line.strip()
        if line:
            line_split = line.split( ',' )
            if line_split[ -1 ] not in classes:
                classes[ line_split[ -1 ] ] = [ ]
            classes[ line_split[ -1 ] ].append( line_split[ 0 ] )

training_samples = [ ]
test_samples = [ ]
for label in classes:
    training_amount = int( ( float( len( classes[ label ] ) ) * float( training_perc ) ) / 100.0 )
    # Create the training set by random sampling
    training_samples.extend( random.sample( classes[ label ], training_amount ) )
    test_samples.extend( [ sample for sample in classes[ label ] if sample not in training_samples ] )

with open( inputfile ) as dataset:
    # skip header
    dataset.readline()
    # read dataset
    for line in dataset:
        line = line.strip()
        if line:
            line_split = line.split( ',' )
            sample = line_split[ 0 ]
            label = line_split[ -1 ]
            line_split = line_split[ 1: -1 ]
            for idx in range( len( line_split ) ):
                if line_split[ idx ] == "?":
                    line_split[ idx ] = 0.0
                else:
                    line_split[ idx ] = float( line_split[ idx ] )
            if sample in training_samples:
                trainData.append( line_split )
                trainLabels.append( label )
            elif sample in test_samples:
                testData.append( line_split )
                testLabels.append( label )

print( len( features ) )
print( len( trainData ) )
print( len( trainLabels ) )
print( len( testData ) )
print( len( testLabels ) )

pickledata = ( features, trainData, trainLabels, testData, testLabels )
with open( os.path.join( basepath, '{}.pkl'.format(inputname)), 'wb' ) as f:
    pickle.dump( pickledata, f )
