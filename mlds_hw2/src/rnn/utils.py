import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

def loadDataset(filename, totalSetNum):
    print '... loading data'
    # Load the dataset
    f = open(filename, 'rb')
    datasets = cPickle.load(f)
    f.close()
    return datasets

def loadPkl(filename):
    f = open(filename, 'rb')
    temp = cPickle.load(f)
    f.close()
    return temp

def namepick(name):
    part = name.split('_')
    return (part[0] + '_' + part[1]), int(part[2])

def findEndIndxGroup(dataName):
    endIndxGroup = []
    curName, number = namepick(dataName[0])
    dataLen = len(dataName)
    for i in xrange(dataLen):
        nextName, nextNum = namepick(dataName[i])
        if(curName != nextName):
            endIndxGroup.append(i)
            curName = nextName
    endIndxGroup.append(dataLen)
    return endIndxGroup

def makePkl(pkl, filename):
    f = open(filename, 'wb')
    cPickle.dump(pkl, f, protocol=2)
    f.close()


def readFile(filename):
    f = open(filename, 'r')
    name = []
    label = []
    for i in f:
        part = i.split(',')
        name.append(part[0])
        label.append(part[1])
    f.close()
    return name, label

def readSetting(filename):
    f = open(filename, 'r')
    title = []
    parameter = []
    for i in f:
        part = i.split(':')
        title.append(part[0])
        parameter.append(part[1])
    f.close()
    return title, parameter

def pickResultFilename(resultFilename):
    tmp = resultFilename.split('/')
    return tmp[len(tmp)-1]

# For "data_prepare/2_pick_value"
# It will return speaker interval and the total frames of this seakper. 
# Return format is the list of " (start index, end index, speakname) "
def findSpeakerInterval(y):
    prevName = y[0][0]
    start = 0 
    end = 0
    maleSpeakerInterval = []
    femaleSpeakerInterval = []
    totalMaleFrameNum = 0
    totalFemaleFrameNum = 0
    for i in xrange(1, len(y)):
        curName = y[i][0]
        if (prevName != curName):
            end = i - 1
            if prevName[0] == 'm':
                maleSpeakerInterval.append((start, end, end - start + 1, prevName))
                totalMaleFrameNum += (end - start + 1)
            elif prevName[0] == 'f':
                femaleSpeakerInterval.append((start, end, end - start + 1, prevName))
                totalFemaleFrameNum += (end - start + 1)
            start = i
        prevName = curName
    end = len(y) - 1
    if prevName[0] == 'm':
        maleSpeakerInterval.append((start, end, end - start + 1, prevName))
        totalMaleFrameNum += (end - start + 1)
    elif prevName[0] == 'f':
        femaleSpeakerInterval.append((start, end, end - start + 1, prevName))
        totalFemaleFrameNum += (end - start + 1)
    return maleSpeakerInterval, totalMaleFrameNum, femaleSpeakerInterval, totalFemaleFrameNum

# For writing a sorted ark file
def writeArkFile(x, fileName):
    f = open(filename, 'w')
    for i in xrange(len(x)):
        tmp = x[i][0] + '_' + x[i][1] + '_' + str(x[i][2]) + ' ' + x[i][3] + '\n'
        f.write(tmp)
    f.close()

# For writing a sorted label file
def writeLabelFile(y, fileName):
    f = open(filename, 'w')
    for i in xrange(len(y)):
        tmp = y[i][0] + '_' + y[i][1] + '_' + str(y[i][2]) + ',' + x[i][3] + '\n'
        f.write(tmp)
    f.close()

