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

def makeModelPkl(model, modelfilename, P):
    modelPkl = [P, params]
    utils.makePkl(modelPkl, modelFilename)

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

def readFile2(filename):
    f = open(filename, 'r')
    title = []
    parameter = []
    for i in f:
        part = i.split(':')
        title.append(part[0])
        parameter.append(part[1])
    f.close()
    return title, parameter

def readModelPkl(filename):
    f = open(filename, 'rb')
    bestModel = cPickle.load(f)
    return bestModel

def pickResultFilename(resultFilename):
    tmp = resultFilename.split('/')
    return tmp[len(tmp)-1]

def findSpeakerInterval(speakerNameList):
    prevName = speakerNameList[0][1]
    start = 0
    end = 0
    speakerInterval = []
    speakerNameListLen = len(speakerNameList)
    for i in xrange(speakerNameListLen):
        curName = speakerNameList[i][1]
        if (prevName != curName):
            end = i
            name = prevName
            speakerInterval.append((start, end, name))
            start = i
        prevName = curName
    speakerInterval.append((start, speakerNameListLen, prevName))
    return speakerInterval

