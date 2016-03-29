import os
import sys
import timeit
import random
import numpy
import theano
import theano.tensor as T

import dnnUtils
import globalParam
from dnnUtils import Parameters
from dnnArchitecture import HiddenLayer, OutputLayer, DNN

DEBUG = False

parameterFilename = sys.argv[1]
numpy.set_printoptions(threshold=numpy.nan) # for print numpy array

def trainDNN(datasets, P):

    trainSetX, trainSetY, trainSetName = datasets[0]
    validSetX, validSetY, validSetName = datasets[1]
    sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = dnnUtils.sharedDataXY(trainSetX, trainSetY)
    sharedValidSetX, sharedValidSetY, castSharedValidSetY = dnnUtils.sharedDataXY(validSetX, validSetY)

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()

    # For create a new model
    dummyParams = [None] * (2 * (P.dnnDepth + 1))
    
    # Build the DNN object for training
    classifier = DNN( input = dnnUtils.splicedX(sX, idx, P.spliceWidth), params = dummyParams, P = P, DROPOUT = True )
    
    # Build the DNN object for Validation
    predicter = DNN( input = dnnUtils.splicedX(sX, idx, P.spliceWidth), P = P, params = dummyParams )
    
    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(dnnUtils.splicedY(sY,idx)) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
   
    # Global parameters setting
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()
    
    grads = [T.grad(cost, param) for param in classifier.params]
    myOutputs = [classifier.errors(dnnUtils.splicedY(sY, idx))] + grads + classifier.params + [cost]
    myUpdates = dnnUtils.chooseUpdateMethod(grads, classifier.params, P)

    # Training mode
    trainModel = theano.function( inputs = [idx], outputs = myOutputs, updates = myUpdates, 
                                  givens={sX:sharedTrainSetX, sY:castSharedTrainSetY})

    # Validation model
    validModel = theano.function( inputs = [idx], outputs = predicter.errors(dnnUtils.splicedY(sY, idx)),
                                  givens={sX:sharedValidSetX, sY:castSharedValidSetY})

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,\ttrain,\tvalid,\tcost')

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
    trainCenterIdx = dnnUtils.findCenterIdxList(trainSetY)
    validCenterIdx = dnnUtils.findCenterIdxList(validSetY)
    
    # Total Center Index
    totalTrainSize = len(trainCenterIdx)
    totalValidSize = len(validCenterIdx)
    
    # Make mini-Batch
    trainBatchIdx = dnnUtils.makeBatch(totalTrainSize, P.batchSizeForTrain)
    validBatchIdx = dnnUtils.makeBatch(totalValidSize, 16384)

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        random.shuffle(trainCenterIdx)

        # Training
        trainLosses=[]
        cost = 0
        for i in xrange(len(trainBatchIdx)):
            outputs = trainModel(trainCenterIdx[trainBatchIdx[i][0]:trainBatchIdx[i][1]])
            trainLosses.append(outputs[0])
            cost += outputs[-1]
            # Print parameter value for debug
            if (i == 0 and DEBUG):
                dnnUtils.printGradsParams(outputs[1:], P.dnnDepth)

        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = dnnUtils.getParamsValue(classifier.params)
        dnnUtils.setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(validCenterIdx[validBatchIdx[i][0]:validBatchIdx[i][1]]) for i in xrange(len(validBatchIdx))]
        validFER = numpy.mean(validLosses)

        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
            curEarlyStop = 0
        elif globalParam.lr > 1e-5:
            epoch -= 1
            dnnUtils.setParamsValue(prevModel, classifier.params)

            if curEarlyStop < P.earlyStop:
                curEarlyStop += 1
            elif (curEarlyStop > P.earlyStop) and (P.updateMethod != 'Momentum'):
                globalParam.lr = globalParam.lr * P.learningRateDecay
                print "learning decay!!"

            if P.updateMethod == 'Momentum':
                globalParam.lr = globalParam.lr * P.learningRateDecay
            elif P.updateMethod == 'Adagrad':
                globalParam.gradSqrs = prevGradSqrs
            elif P.updateMethod == 'RMSProp':
                globalParam.sigmaSqrs = prevSigmaSqrs
            print (('====,%i,%f,%f,%f') % (epoch, trainFER * 100, validFER * 100., cost ))
            continue
        else:
            doneLooping = True
            continue

        print (('%i,%f,%f,%f') % (epoch, trainFER * 100, validFER * 100. , cost))

        # Record the Adagrad, RMSProp parameter
        if P.updateMethod == 'Adagrad':
            prevGradSqrs = globalParam.gradSqrs
        if P.updateMethod == 'RMSProp':
            prevSigmaSqrs = globalParam.sigmaSqrs

    # end of training
        
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))

    dnnUtils.clearSharedDataXY(sharedTrainSetX, sharedTrainSetY)
    dnnUtils.clearSharedDataXY(sharedValidSetX, sharedValidSetY)

    return prevModel

def getResult(bestModel, dataset, P, typeName, resultFilename):

    print "...getting result"

    setX, setY, setName = dataset
    sharedSetX, sharedSetY, castSharedSetY = dnnUtils.sharedDataXY(setX, setY)
    
    print "...buliding model"
    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()
    
    # bulid best DNN  model
    predicter = DNN( input = dnnUtils.splicedX(sX, idx, P.spliceWidth), P = P, params = bestModel )
    
    # Center Index
    centerIdx = dnnUtils.findCenterIdxList(setY)
    
    # Total Center Index
    totalCenterIdxSize = len(centerIdx)
    
    # Make mini-Batch
    batchIdx = dnnUtils.makeBatch(totalCenterIdxSize, 16384)
    
    Model = theano.function( inputs = [idx], outputs = [predicter.errors(dnnUtils.splicedY(sY, idx)),predicter.yPred],
                                  givens={sX:sharedSetX, sY:castSharedSetY})
    
    result = dnnUtils.EvalandResult(Model, batchIdx, centerIdx, typeName) 
    
    dnnUtils.writeResult(result, centerIdx, resultFilename, setName)
    
    dnnUtils.clearSharedDataXY(sharedSetX, sharedSetY)

def getProb(bestModel, dataset, probFilename, P):

    print "...getting probability"
    setX, setY, setName = dataset
    sharedSetX, sharedSetY, castSharedSetY = dnnUtils.sharedDataXY(setX, setY)

    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()

    # bulid best DNN model
    predicter = DNN( input = dnnUtils.splicedX(sX, idx, P.spliceWidth), P = P, params = bestModel )

    # Validation model
    Model = theano.function( inputs = [idx], outputs = predicter.p_y_given_x, 
                             givens={sX:sharedSetX, sY:castSharedSetY}, on_unused_input='ignore')
    
    # Center Index
    centerIdx = dnnUtils.findCenterIdxList(setY)

    # Total Center Index
    totalCenterIdxSize = len(centerIdx)
    
    # Make mini-Batch
    batchIdx = dnnUtils.makeBatch(totalCenterIdxSize, 16384)
    
    # Writing Probability
    dnnUtils.writeProb(Model, batchIdx, centerIdx, setName, probFilename)

    dnnUtils.clearSharedDataXY(sharedSetX, sharedSetY)
