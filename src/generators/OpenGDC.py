import sys, os, pickle, random
from pathlib import Path

trainData = [ ]
trainLabels = [ ]
testData = [ ]
testLabels = [ ]

# We just need the training percentage
# The test percentage is 100.0 - training_perc
training_perc = 80.0

program = 'tcga'
tumor = '{}-acc'.format( program )

# valueidx is the id of the column which contains the observation value for a specific experiment
# It starts from 0

# DNA-Methylation 450
datatype = 'methylation_beta_value'
metaabbr = 'mbv'
valueidx = 5

# Gene-expression Quantification FPKM
#datatype = 'gene_expression_quantification'
#metaabbr = 'geq'
#valueidx = 10

# Input files have the same number of features, all in the same order
basepath = '/Users/fabio/GitHub/HD-Classifier/datasets/OpenGDC'
inputname = '{}__{}'.format( tumor, datatype )

bed_basepath = '{}/{}/{}/{}/'.format( basepath, program, tumor, datatype )
metadata_basepath = '{}/{}/{}/clinical_and_biospecimen_supplements/'.format( basepath, program, tumor )

classes = { }
bed_generator = Path( bed_basepath ).glob( '*.bed' )
for filepath in bed_generator:
    sample = '-'.join( os.path.splitext( os.path.basename( str( filepath ) ) )[ 0 ].split( '-' )[ 0: -1 ] )
    metadata_filepath = os.path.join( metadata_basepath, '{}-{}.bed.meta'.format( sample, metaabbr ) )
    platform = None
    label = None
    if os.path.exists( metadata_filepath ):
        with open( metadata_filepath ) as metadata:
            for line in metadata:
                line = line.strip()
                if line:
                    line_split = line.split( '\t' )
                    if line_split[ 0 ] == 'manually_curated__tissue_status':
                        label = line_split[ 1 ]
                    elif line_split[ 0 ] == 'gdc__platform':
                        platform = line_split[ 1 ]
    if ( datatype == 'methylation_beta_value' and platform == 'Illumina Human Methylation 450' ) or \
            not datatype == 'methylation_beta_value' and label:
        if label not in classes:
            classes[ label ] = [ ]
        classes[ label ].append( sample )

# Start processing if more than one class has been detected
if len( classes ) > 1:
    training_samples = [ ]
    test_samples = [ ]
    for label in classes:
        print( '{}: {}'.format( label, len( classes[ label ] ) ) )
        training_amount = int( ( float( len( classes[ label ] ) ) * float( training_perc ) ) / 100.0 )
        # Create the training set by random sampling
        training_samples.extend( random.sample( classes[ label ], training_amount ) )
        test_samples.extend( [ sample for sample in classes[ label ] if sample not in training_samples ] )

    bed_generator = Path( bed_basepath ).glob( '*.bed' )
    for filepath in bed_generator:
        sample = '-'.join( os.path.splitext( os.path.basename( str( filepath ) ) )[ 0 ].split( '-' )[ 0: -1 ] )
        if sample in training_samples or sample in test_samples:
            print( 'Reading {}'.format( sample ) )
            values = [ ]
            with open( str( filepath ) ) as samplefile:
                for line in samplefile:
                    line = line.strip()
                    if line:
                        value = line.split( '\t' )[ valueidx ]
                        if value.strip():
                            values.append( float( value ) )
                        else:
                            values.append( 0.0 )
            trainData.append( values ) if sample in training_samples else testData.append( values )
            for label in classes:
                if sample in classes[ label ]:
                    trainLabels.append( label ) if sample in training_samples else testLabels.append( label )
                    break

    print( len( trainData ) )
    print( len( trainLabels ) )
    print( len( testData ) )
    print( len( testLabels ) )

    pickledata = ( trainData, trainLabels, testData, testLabels )
    with open( os.path.join( basepath, '{}.pkl'.format(inputname)), 'wb' ) as f:
        pickle.dump( pickledata, f )
else:
    print( '{} classes found'.format( len( classes ) ) )
    sys.exit( 1 )
