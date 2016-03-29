import sys
import gzip
import random
import cPickle
import numpy as np
import utils
from operator import itemgetter, attrgetter

random.seed(1234)
dim = int(sys.argv[1])
dataPath = sys.argv[2]
pickedValidDirPath = '../../fbank_valid/'
trainArkFilename   = pickedValidDirPath + 'train.ark'
trainLabelFilename = pickedValidDirPath + 'train.lab'
validArkFilename   = pickedValidDirPath + 'valid.ark'
validLabelFilename = pickedValidDirPath + 'valid.lab'
testArkFilename    = dataPath + '/fbank/test.ark'
outputPklFilename  = '../../pkl/fbank_' + str(dim) + '_dataset_without_preprocessing.pkl'

def countLineNum(fileArkName):
    f = open(fileArkName, 'rb')
    num = 0
    for i in f:
        num += 1
    f.close()
    return num

def covertData(fileArkName, LineNum, fileLabelName = None, existY = True):
    dataX = []
    dataY = []
    npDataX = np.zeros((LineNum, dim))
    npDataY = np.zeros((LineNum,))
    dataName = [''] *  LineNum
    fileForX = open(fileArkName, 'rb')
    for curLine in fileForX:
        curLine = curLine.strip()
        curLine = curLine.split()
        name, number = utils.namepick(curLine[0])
        feature = []
        for i in xrange(dim):
            feature.append(float(curLine[i+1]))
        dataX.append([name, number,feature])
    if fileLabelName is not None:
        dataX = sorted(dataX, key=itemgetter(0,1))  
    
    if existY:
        fileForY = open(fileLabelName, 'rb')
        for curLine in fileForY:
            curLine = curLine.strip()
            curLine = curLine.split(',')
            name, number = utils.namepick(curLine[0])
            label = int(curLine[1])
            dataY.append([name, number, label])
        if existY:
            dataY = sorted(dataY, key=itemgetter(0,1))
    else:
        dataY = [[0,'',0]] * len(dataX)
    dataset = []
    for i in xrange(len(dataX)):
        npDataX[i] = dataX[i][2]
        npDataY[i] = dataY[i][2]
        dataName[i] = dataX[i][0] + '_' + str(dataX[i][1])

    return npDataX, npDataY, dataName

if __name__ == '__main__':
    print '... covert training set'
    trainLineNum = countLineNum(trainArkFilename)
    trainSet = covertData(fileArkName = trainArkFilename, fileLabelName = trainLabelFilename, LineNum = trainLineNum)
    print '... covert valid set'
    validLineNum = countLineNum(validArkFilename)
    validSet = covertData(fileArkName = validArkFilename, fileLabelName = validLabelFilename, LineNum = validLineNum)
    print '... covert test set'
    testLineNum = countLineNum(testArkFilename)
    testSet = covertData(fileArkName = testArkFilename, LineNum = testLineNum, existY = False)
    print '... make pkl file'
    utils.makePkl([trainSet, validSet, testSet], outputPklFilename)
