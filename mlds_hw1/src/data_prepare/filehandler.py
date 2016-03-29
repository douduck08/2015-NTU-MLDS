import os
import sys
import cPickle
import numpy

InputNumber = 0
InputSize = 39

def readPkl(filename):
    f = open(filename, 'rb')
    dataset = cPickle.load(f)
    f.close()
    dataset = numpy.asarray(dataset)
    return dataset

def writePkl(data, filename):
    f = open(filename, 'wb')
    cPickle.dump(data, f, protocol=2)
    f.close()

def nameSplit(name):
    part = name.split('_')
    return (part[0] + '_' + part[1]), int(part[2])

def readOriginFile(inputFile, outputFile = None):
    dataX = []
    dataY = []
    fileStream = open(inputFile, 'rb')
    for row in fileReader:
        row = row.strip()
        row = row.split()
        name, number = nameSplit.(row[0])
        feature = []
        for i in xrange(InputSize):
            feature.append(float(row[i+1]))
        dataX.append([name, number, feature])
    dataX = sorted(dataX, key=itemgetter(0,1))
    InputNumber = len(dataX)

    if outputFile is not None:
        fileStream = open(outputFile, 'rb')
        for row in fileReader:
            row = row.strip()
            row = row.split()
            name, number = nameSplit.(row[0])
            label = int(row[1])
            dataY.append([name, number, label])
        dataY = sorted(dataY, key=itemgetter(0,1))

    npDataX = numpy.zeros((InputNumber, InputSize))
    npDataY = numpy.zeros((InputNumber, ))
    dataName = [''] * InputNumber

    for i in xrange(InputNumber):
        npDataX[i] = dataX[i][2]
        npDataY[i] = dataY[i][2]
        dataName[i] = dataX[i][0] + '_' + str(dataX[i][1])

    return npDataX, npDataY, dataName

if __name__ == '__main__':
