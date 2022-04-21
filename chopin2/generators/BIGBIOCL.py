import os, pickle, random

basepath = "/Users/fabio/GitHub/HD-Classifier/datasets/BIGBIOCL/thca_dnameth/"
inputname = "thca_dnameth_NOCONTROL"
inputfile = os.path.join( basepath, "{}.csv".format(inputname) )

features = list()
classes = dict()
with open(inputfile) as dataset:
    # skip header
    features = dataset.readline().strip().split(',')[1:-1]
    # read dataset
    for line in dataset:
        line = line.strip()
        if line:
            line_split = line.split(',')
            if line_split[-1] not in classes:
                classes[line_split[-1]] = list()
            classes[line_split[-1]].append(line_split[0])

allData = list()
allLabels = list()
with open( inputfile ) as dataset:
    # skip header
    dataset.readline()
    # read dataset
    for line in dataset:
        line = line.strip()
        if line:
            line_split = line.split(',')
            sample = line_split[0]
            label = line_split[-1]
            line_split = line_split[1:-1]
            for idx in range(len(line_split)):
                if line_split[idx] == "?":
                    line_split[idx] = 0.0
                else:
                    line_split[idx] = float(line_split[idx])
            allData.append(line_split)
            allLabels.append(label)

print("Features: {}".format(len(features)))
print("Data: {}".format(len(allData)))
print("Labels: {}".format(len(set(allLabels))))

pickledata = (features, allData, allLabels)
with open(os.path.join(basepath, '{}.pkl'.format(inputname)), 'wb') as f:
    pickle.dump(pickledata, f)
