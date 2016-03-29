import utils
import numpy as np
import theano
import theano.tensor as T
import updateMethod
import theano.typed_list

def sharedDataXY(dataX, dataY, borrow=True):
    sharedX = theano.shared(np.asarray(dataX, dtype=theano.config.floatX), borrow=True)
    #TODO does't work in GPU for sharedY
    sharedY = theano.shared(np.asarray(dataY, dtype=theano.config.floatX), borrow=True)
    return [sharedX, sharedY, T.cast(sharedY,'int32')]

def clearSharedDataXY(sharedX, sharedY):
    sharedX.set_value([[]])
    sharedY.set_value([])

def chooseUpdateMethod(grads, params, P):
    if P.updateMethod == 'Momentum':
        return updateMethod.Momentum(grads, params, P.momentum)
    if P.updateMethod == 'RMSProp':
        return updateMethod.RMSProp(grads, params)
    if P.updateMethod == 'Adagrad':
        return updateMethod.Adagrad(grads, params)

# GP means gradient and parameter(W and b).
def printGradsParams(GP, dnnDepth):
    for i in xrange(0, (2 * dnnDepth+1), 2):
        print ( '================ Layer %d ================' % (i/2 + 1))
        printNpArrayMeanStdMaxMin("GW", GP[i])
        printNpArrayMeanStdMaxMin("Gb", GP[i+1])
        printNpArrayMeanStdMaxMin("W ", GP[i+dnnDepth])
        printNpArrayMeanStdMaxMin("b ", GP[i+dnnDepth+1])

def printNpArrayMeanStdMaxMin(name, npArray):
    print(" #%s \t mean = %f \t std = %f \t max = %f \t min = %f" % (name, np.mean(npArray), np.std(npArray), np.amax(npArray), np.amin(npArray) ))

def EvalandResult(Model, batchIdx, centerIdx, modelType):
    result = []
    Losses = []
    for i in xrange(len(batchIdx)):
        thisLoss, thisResult = Model(centerIdx[batchIdx[i][0]:batchIdx[i][1]])
        result += thisResult.tolist()
        Losses.append(thisLoss)
    FER = np.mean(Losses)
    print ((modelType + ' FER,%f') % (FER * 100))
    return result

def writeResult(result, centerIdx, filename, setNameList):
    f = open(filename, 'w')
    for i in xrange(len(result)):
        f.write(setNameList[centerIdx[i]] + ',' + str(result[i]) + '\n')
    f.close()

def writeProb(Model, batchIdx, centerIdx, nameList, filename):
    f = open(filename, 'w')
    k = 0
    for i in xrange(len(batchIdx)):
        tmpProb = Model(centerIdx[batchIdx[i][0]:batchIdx[i][1]]).tolist()
        for j in xrange(batchIdx[i][1]-batchIdx[i][0]):
            f.write(nameList[centerIdx[j+k]] + ' ' + " ".join(map(str, tmpProb[j])) + '\n')
        k += batchIdx[i][1] - batchIdx[i][0]
    f.close()
"""
def writeProb(prob, filename, setNameList):
    print "123"
    f = open(filename, 'w')
    for i in xrange(len(prob)):
        f.write( setNameList[i] + ' ' + " ".join(map(str, prob[i])) + '\n')
    f.close()
"""
def makeBatch(totalSize, batchSize = 32):
    numBatchSize = totalSize / batchSize
    indexList = [[i * batchSize, (i + 1) * batchSize] for i in xrange(numBatchSize)]
    indexList.append([numBatchSize * batchSize, totalSize])
    return indexList

def getParamsValue(nowParams):
    params = []
    for i in xrange(len(nowParams)):
        params.append(nowParams[i].get_value())
    return params

def setParamsValue(preParams, nowParams):
    for i in xrange(len(preParams)):
        nowParams[i].set_value(preParams[i])

def Dropout(rng, input, inputNum, D = None, dropoutProb = 1):
    D_values = np.asarray(
              rng.binomial( size = (inputNum,), n = 1, p = dropoutProb ),
              dtype=theano.config.floatX )
    D = theano.shared( value=D_values, name='D', borrow=True )
    return input * D

def findCenterIdxList(dataY):
    spliceIdxList = []
    for i in xrange(len(dataY)):
        if dataY[i] == -1:
            continue
        else:
            spliceIdxList.append(i)
    return spliceIdxList

def splicedX(x, idx, spliceWidth):
    return T.concatenate([ (T.stacklists([ (x[j+i] / (abs(i)+1))  for j in [idx] ])) for i in xrange(-spliceWidth, spliceWidth+1)])
#return T.concatenate([ x[idx+i] for i in xrange(-spliceWidth, spliceWidth+1)])

# Not really spliced Y data
def splicedY(y, idx):    
    return T.concatenate([y[i] for i in [idx]])
    
class Parameters(object):
    def __init__(self, filename):
       title, parameter           = utils.readSetting(filename)
       self.momentum              = float(parameter[title.index('momentum')])
       self.dnnWidth              = int(parameter[title.index('width')])
       self.dnnDepth              = int(parameter[title.index('depth')])
       self.spliceWidth           = int(parameter[title.index('spliceWidth')])
       self.batchSizeForTrain     = int(parameter[title.index('batchSize')])
       self.learningRate          = float(parameter[title.index('learningRate')])
       self.learningRateDecay     = float(parameter[title.index('learningRateDecay')])
       self.dropoutHiddenProb     = float(parameter[title.index('hiddenProb')])
       self.dropoutInputProb      = float(parameter[title.index('inputProb')])
       self.datasetFilename       = parameter[(title.index('dataSetFilename'))].strip('\n')
       self.datasetType           = parameter[(title.index('dataSetType'))].strip('\n')
       self.updateMethod          = parameter[(title.index('updateMethod'))].strip('\n')
       self.maxEpoch              = int(parameter[title.index('maxEpoch')])
       self.inputDimNum           = int(parameter[title.index('inputDimNum')])
       self.outputPhoneNum        = int(parameter[title.index('outputPhoneNum')])
       self.seed                  = int(parameter[title.index('seed')])
       self.earlyStop             = int(parameter[title.index('earlyStop')])
       self.L1Reg                 = float(parameter[title.index('L1Reg')])
       self.L2Reg                 = float(parameter[title.index('L2Reg')])
       self.outputFilename = (str(self.datasetType) + '_' + (str(self.updateMethod))
                              + '_m_' + str(self.momentum)
                              + '_dw_'+ str(self.dnnWidth)
                              + '_dd_'+ str(self.dnnDepth)
                              + '_sw_'+ str(self.spliceWidth)
                              + '_b_' + str(self.batchSizeForTrain)
                              + '_lr_'+ str(self.learningRate)
                              + '_lrd_' + str(self.learningRateDecay)
                              + '_di_'+ str(self.dropoutInputProb)
                              + '_dh_'+ str(self.dropoutHiddenProb) )
       self.bestModelFilename   = '../model/' + self.outputFilename
       self.trainProbFilename   = '../prob/train/' +self.outputFilename + '.ark'
       self.validProbFilename   = '../prob/valid/' + self.outputFilename + '.ark'
       self.testProbFilename   = '../prob/test/' + self.outputFilename + '.ark'
       self.testResultFilename  = '../result/test_result/' + self.outputFilename + '.csv'
       self.validResultFilename = '../result/valid_result/' + self.outputFilename + '.csv'
       self.validSmoothedResultFilename = '../result/smoothed_valid_result/' + self.outputFilename + '.csv' 
       self.testSmoothedResultFilename  = '../result/smoothed_test_result/' + self.outputFilename + '.csv' 
       self.logFilename = '../log/' + self.outputFilename + '.log'
       self.rng = np.random.RandomState(self.seed)


