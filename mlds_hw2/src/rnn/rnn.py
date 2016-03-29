import os
import sys
import timeit
import random
import numpy
import theano
import theano.tensor as T
import filler
import rnnUtils
import globalParam
from rnnUtils import Parameters
from rnnArchitecture import HiddenLayer, OutputLayer, RNN

DEBUG = False

parameterFilename = sys.argv[1]
numpy.set_printoptions(threshold=numpy.nan) # for print numpy array

def trainDNN(datasets, P):

    trainSetX, trainSetY, trainSetName, trainMask = filler.fillerCore(datasets[0])
    validSetX, validSetY, validSetName, validMask = filler.fillerCore(datasets[1])
    sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = rnnUtils.sharedDataXY(trainSetX, trainSetY)
    sharedValidSetX, sharedValidSetY, castSharedValidSetY = rnnUtils.sharedDataXY(validSetX, validSetY)

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    idx = T.iscalar('i')
    x = T.matrix(dtype=theano.config.floatX)
    y = T.ivector()

    # For create a new model
    dummyParams = [None] * (4 * (P.rnnDepth) + 2)

    # Build the RNN object for training
    classifier = RNN( input = x, params = dummyParams, P = P)

    # Build the DNN object for Validation
    predicter = RNN( input = x, P = P, params = dummyParams )

    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(y) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )

    # Global parameters setting
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()

    grads = [T.grad(cost, param) for param in classifier.params]
    myOutputs = [classifier.errors(y)] + grads + classifier.params
    myUpdates = rnnUtils.chooseUpdateMethod(grads, classifier.params, P)

    # Training mode
    trainModel = theano.function( inputs = [idx], outputs = myOutputs, updates = myUpdates,
                                  givens = {x:sharedTrainSetX[idx], y:castSharedTrainSetY[idx]})

    # Validation model
    validModel = theano.function( inputs = [idx], outputs = predicter.errors(y),
                                  givens = {x:sharedValidSetX[idx], y:castSharedValidSetY[idx]})

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,    train,    valid')

    # Training parameter
    epoch = 0
    curEarlyStop = 0
    prevModel = None
    nowModel = None
    doneLooping = False
    prevFER = numpy.inf
    random.seed(P.seed)

    # Adagrad, RMSProp
    prevGradSqrs = None
    prevSigmaSqrs = None

    # Center Index
#trainCenterIdx = rnnUtils.findCenterIdxList(trainSetY)
#validCenterIdx = rnnUtils.findCenterIdxList(validSetY)

    # Total Center Index
#totalTrainSize = len(trainCenterIdx)
#totalValidSize = len(validCenterIdx)

    totalTrainSentSize = len(trainSetX)
    totalValidSentSize = len(validSetX)

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

#random.shuffle(trainCenterIdx)

        # Training
        trainLosses=[]
        for i in xrange(totalTrainSentSize):
            outputs = trainModel(i)
            trainLosses.append(outputs[0])

            # Print parameter value for debug
            if (i == 0 and DEBUG):
                rnnUtils.printGradsParams(outputs[1:], P.rnnDepth)

        # Evaluate training FER
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = rnnUtils.getParamsValue(classifier.params)
        rnnUtils.setParamsValue(nowModel, predicter.params)

        # Evaluate validation FER
        validLosses = [validModel(i) for i in xrange(totalValidSentSize)]
        validFER = numpy.mean(validLosses)
        prevModel = nowModel
        """
        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
            curEarlyStop = 0
        else:
            if curEarlyStop < P.earlyStop:
                epoch -= 1
                rnnUtils.setParamsValue(prevModel, classifier.params)
                print (('====,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                curEarlyStop += 1
                if P.updateMethod == 'Momentum':
                    globalParam.lr = globalParam.lr * P.learningRateDecay
                elif P.updateMethod == 'Adagrad':
                    globalParam.gradSqrs = prevGradSqrs
                elif P.updateMethod == 'RMSProp':
                    globalParam.sigmaSqrs = prevSigmaSqrs
                continue
            else:
                doneLooping = True
                continue
        """
        print (('%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))

        # Record the Adagrad, RMSProp parameter
        if P.updateMethod == 'Adagrad':
            prevGradSqrs = globalParam.gradSqrs
        if P.updateMethod == 'RMSProp':
            prevSigmaSqrs = globalParam.sigmaSqrs

    # end of training

    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))

#    rnnUtils.clearSharedDataXY(sharedTrainSetX, sharedTrainSetY)
#    rnnUtils.clearSharedDataXY(sharedValidSetX, sharedValidSetY)

    return prevModel

def getResult(bestModel, datasets, P):

    print "...getting result"

    validSetX, validSetY, validSetName = datasets[1]
    testSetX, testSetY, testSetName = datasets[2]
    sharedValidSetX, sharedValidSetY, castSharedValidSetY = rnnUtils.sharedDataXY(validSetX, validSetY)
    sharedTestSetX, sharedTestSetY, castSharedTestSetY = rnnUtils.sharedDataXY(testSetX, testSetY)

    print "...buliding model"
    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()

    # bulid best DNN  model
    predicter = RNN( input = sX, P = P, params = bestModel )

    # Center Index
    validCenterIdx = rnnUtils.findCenterIdxList(validSetY)
    testCenterIdx = rnnUtils.findCenterIdxList(testSetY)

    # Total Center Index
#totalValidSize = len(validCenterIdx)
#totalTestSize = len(testCenterIdx)

    # Make mini-Batch
#validBatchIdx = rnnUtils.makeBatch(totalValidSize, 16384)
#testBatchIdx = rnnUtils.makeBatch(totalTestSize, 16384)
    totalTestSentSize = len(testSetX)
    totalValidSentSize = len(validSetX)

    # Validation model
    validModel = theano.function( inputs = [idx], outputs = [predicter.errors(sY),predicter.yPred],
                                  givens={sX:sharedValidSetX, sY:castSharedValidSetY})

    # bulid test model
    testModel = theano.function( inputs = [idx], outputs = [predicter.errors(sY),predicter.yPred],
                                  givens={sX:sharedTestSetX, sY:castSharedTestSetY})

    validResult = rnnUtils.EvalandResult(validModel, totalValidSize, 'valid')
    testResult = rnnUtils.EvalandResult(testModel, totalTestSize, 'test')

    rnnUtils.writeResult(validResult, P.validResultFilename, validSetName)
    rnnUtils.writeResult(testResult, P.testResultFilename, testSetName)

    rnnUtils.clearSharedDataXY(sharedTestSetX, sharedTestSetY)
    rnnUtils.clearSharedDataXY(sharedValidSetX, sharedValidSetY)


def getProb(bestModel, dataset, probFilename):

    print "...getting probability"
    # For getting prob
    setX, setY, setName = dataset
    sharedSetX, sharedSetY, castSharedSetY = rnnUtils.sharedDataXY(setX, setY)

    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()

    # bulid best DNN model
    predicter = DNN( input = rnnUtils.splicedX(sX, idx), P = P, params = bestModel )

    Model = theano.function( inputs = [idx], outputs = predicter.p_y_given_x,
                                  givens={sX:sharedSetX, sY:castSharedSetY}, on_unused_input='ignore')

    # Center Index
    centerIdx = rnnUtils.findCenterIdxList(setY)

    # Total Center Index
    totalSize = len(centerIdx)

    # Make mini-Batch
    batchIdx = rnnUtils.makeBatch(totalSize, 16384)

    # Writing Probability
    rnnUtils.writeProb(Model, batchIdx, centerIdx, setName, probFilename)

    rnnUtils.clearSharedDataXY(sharedSetX, sharedSetY)
