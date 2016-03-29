import os
import sys
import cPickle
import gzip
import numpy

ALPHA = 1

def loadPkl(filename):
    f = open(filename, 'rb')
    temp = cPickle.load(f)
    f.close()
    return temp

def makePkl(pkl, filename):
    f = open(filename, 'wb')
    cPickle.dump(pkl, f, protocol=2)
    f.close()

def mergeValidSet(datasets):
    trainSetX, trainSetY, trainSetName = datasets[0]
    validSetX, validSetY, validSetName = datasets[1]
    setX = numpy.vstack([trainSetX, validSetX])
    setY = numpy.concatenate((trainSetY, validSetY))
    setName = trainSetName + validSetName
    return setX, setY, setName

def getTestSet(datasets):
    return datasets[2]


def softmax(setX):
    maxX = numpy.max(setX, axis=1)
    absMaxX = numpy.abs(maxX)
    absMaxX = numpy.reshape(absMaxX, (absMaxX.shape[0], 1))
    expX = numpy.exp( ALPHA * setX  / absMaxX)
    expXsum = numpy.sum(expX, axis=1)
    expXsum = numpy.reshape(expXsum, (expXsum.shape[0], 1))
    setX = expX / expXsum
    return setX

if __name__ == '__main__':
    print "Please put pkl files into './pkl_file' dir, and press enter to continue."
    ch = sys.stdin.read(1)
    fileList = []
    for file in os.listdir("./pkl_file"):
        if file.endswith(".pkl"):
            fileList.append(file)
    fileCount = len(fileList)
    print "Found the files below: (totally have " + str(fileCount)
    print fileList

    softmaxFlag = False
    while True:
        print "\nDo you want to do Softmax on data? (yes/no): ",
        userinput = sys.stdin.readline()
        if userinput == 'yes\n':
            print "Set Softmax Flag = True"
            softmaxFlag = True
            break
        elif userinput == 'no\n':
            print "Set Softmax Flag = False"
            softmaxFlag = False
            break

    averageMethod = 1
    while True:
        print "\nChoose the ensemble method below,"
        print "1)Arithmetic mean, 2)Geometric mean: ",
        userinput = sys.stdin.readline()
        if userinput == '1\n':
            print "Set ensemble method = Arithmetic mean"
            averageMethod = 1
            break
        elif userinput == '2\n':
            print "Set ensemble method = Geometric mean"
            averageMethod = 2
            break

    # print "Setting"
    # userinput = sys.stdin.read(1)
    # print userinput

    totalDataset = [[], [], []]
    totalTestDataset  = [[], [], []]
    flag = True
    print "\n=== Start ensemble ... ==="
    for filename in fileList:
        dataset = loadPkl("./pkl_file/" + filename)
        testDataset = getTestSet(dataset)
        dataset = mergeValidSet(dataset)
        print "The shape of '" + filename + "' = " + str(dataset[0].shape)

        if softmaxFlag:
            setX = softmax(dataset[0])
            testSetX = softmax(testDataset[0])
        else:
            setX = dataset[0]
            testSetX = testDataset[0]

        if flag:
            # print dataset[0][0]
            flag = False
            if averageMethod == 1:
                totalDataset[0] = setX
                totalTestDataset[0] = testSetX
            if averageMethod == 2:
                totalDataset[0] = numpy.power(setX, 2)
                totalTestDataset[0] = numpy.power(testSetX, 2)
            totalDataset[1] = dataset[1]
            totalDataset[2] = dataset[2]
            totalTestDataset[2] = testDataset[2]
        else:
            # print dataset[0][0]
            if averageMethod == 1:
                totalDataset[0] += dataset[0]
                totalTestDataset[0] += testSetX
            if averageMethod == 2:
                totalDataset[0] += numpy.power(setX, 2)
                totalTestDataset[0] += numpy.power(testSetX, 2)

    print "The shape of the Final Sum Data = " + str(totalDataset[0].shape)
    if averageMethod == 1:
        totalDataset[0] = totalDataset[0] / fileCount
        totalTestDataset[0] = totalTestDataset[0] / fileCount
    if averageMethod == 2:
        totalDataset[0] = totalDataset[0] / fileCount
        totalDataset[0] = numpy.sqrt(totalDataset[0])
        totalTestDataset[0] = totalTestDataset[0] / fileCount
        totalTestDataset[0] = numpy.sqrt(totalTestDataset[0])
    # print totalDataset[0][0]

    makePkl([totalDataset, [], totalTestDataset], "ensembled_data.pkl")






