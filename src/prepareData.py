import os
import pickle

trainData = [ ]
trainLabels = [ ]
testData = [ ]
testLabels = [ ]

# test size = 20%, the remaining portion is for training (80%)
# ( grep "tumoral" inputfile.csv | wc -l ) * 20 / 100
test_tumoral = 103 # number of tumoral experiments in test (20%)
# ( grep "normal" inputfile.csv | wc -l ) * 20 / 100
test_normal = 11   # number of normal experiments in test (20%)

basepath = "/Users/fabio/GitHub/HD-Classifier/datasets/thca_dnameth/"
inputname = "thca_dnameth_NOCONTROL"
inputfile = os.path.join( basepath, "{}.csv".format(inputname) )

firstline = True
with open( inputfile ) as m:
    for line in m:
        line = line.strip()
        if line:
            if firstline:
                firstline = False
            else:
                line_split = line.split(",")
                label = line_split[-1]
                line_split = line_split[1:-1]
                for d in range(len(line_split)):
                    if line_split[d] == "?":
                        line_split[d] = 0.0
                    else:
                        line_split[d] = float(line_split[d])
                if label == "tumoral":
                    if test_tumoral == 0:
                        trainData.append(line_split)
                        trainLabels.append(label)
                    else:
                        testData.append(line_split)
                        testLabels.append(label)
                        test_tumoral -= 1
                else:
                    if test_normal == 0:
                        trainData.append(line_split)
                        trainLabels.append(label)
                    else:
                        testData.append(line_split)
                        testLabels.append(label)
                        test_normal -= 1

print(len(trainData))
print(len(trainLabels))
print(len(testData))
print(len(testLabels))

pickledata = ( trainData, trainLabels, testData, testLabels )
with open( os.path.join( basepath, '{}.pkl'.format(inputname)), 'wb' ) as f:
    pickle.dump( pickledata, f )
