from dnnUtils import Dropout
import theano.tensor as T
import theano
import numpy

class FirstHiddenLayer(object):
    def __init__(self, rng, input, inputNum, outputNum, dnnWidth, spliceWidth, W = None, b = None, dropoutProb = 1.0, DROPOUT = False):
        if DROPOUT == True:
            self.input = Dropout( rng = rng, input = input, inputNum = inputNum, dropoutProb = dropoutProb )
        else:
            self.input = input * dropoutProb
        self.input = input
        if W is None:
            W_values = rng.uniform( low = -numpy.sqrt(1./(inputNum+outputNum)), high = numpy.sqrt(1./(inputNum+outputNum)),
            size = (spliceWidth*2 + 1, inputNum/(spliceWidth*2 + 1), outputNum) ).astype( dtype=theano.config.floatX )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if b is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )

        self.W = W
        self.b = b
        
        #z = T.sum(T.batched_dot(self.input, self.W),axis=2) + self.b
        z = T.tensordot(self.input, self.W, [[0,2],[0,1]]) + self.b

        # Maxout
        zEven = z[:,0::2]
        zOld = z[:,1::2]
        self.output = (T.maximum(zOld, zEven)).astype(dtype=theano.config.floatX)
        """
        zT= z.T
        self.output = T.maximum(zT[0:dnnWidth/2],zT[dnnWidth/2:]).T
        """
        # parameters of the model
        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, rng, input, inputNum, outputNum, dnnWidth, W = None, b = None, dropoutProb = 1.0, DROPOUT = False):
        if DROPOUT == True:
            self.input = Dropout( rng = rng, input = input, inputNum = inputNum, dropoutProb = dropoutProb )
        else:
            self.input = input * dropoutProb
        if W is None:
            W_values = rng.uniform( low = -numpy.sqrt(1./(inputNum+outputNum)), high = numpy.sqrt(1./(inputNum+outputNum)),
            size = (inputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if b is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )
        self.W = W
        self.b = b
        
        z = T.dot(self.input, self.W) + self.b
        
        # Maxout
        zEven = z[:, 0::2]
        zOld = z[:, 1::2]
        self.output = (T.maximum(zOld, zEven)).astype(dtype=theano.config.floatX)
        """
        zT= z.dimshuffle(1,0)
        self.output = (T.maximum(zT[0:dnnWidth/2],zT[dnnWidth/2:]).dimshuffle(1,0)).astype(dtype=theano.config.floatX)
        """
        # parameters of the model
        self.params = [self.W, self.b]

class OutputLayer(object):
    def __init__(self, input, inputNum, outputNum, rng, W = None, b = None):
        if W is None:
            W_values = rng.uniform( low = -numpy.sqrt(6./(inputNum+outputNum)), high = numpy.sqrt(6./(inputNum+outputNum)),
                                    size = (inputNum, outputNum) ).astype(dtype=theano.config.floatX )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow=True )

        if b is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow=True )
        self.W = W
        self.b = b
        
        z = T.dot(input, self.W) + self.b
        
        # Softmax
        maxZ = T.max(z, axis=1).astype(dtype = theano.config.floatX)
        absMaxZ = T.abs_(maxZ)
        absMaxZ = T.reshape(absMaxZ, (absMaxZ.shape[0], 1))
        expZ = T.exp(z*10  / absMaxZ)
        expZsum = T.sum(expZ, axis=1).astype(dtype = theano.config.floatX)
        expZsum = T.reshape(expZsum, (expZsum.shape[0],1))
        self.p_y_given_x = (expZ / expZsum)
        
        # Find larget y_i
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
    
    # Cross entropy
    def crossEntropy(self, y):
        return -T.mean( (T.log(self.p_y_given_x)).clip(-1e6, 1e6)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # Check y and y_pred dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # Check if y is the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class DNN(object):
    def __init__(self, input, P, params = None, DROPOUT = False):
        # Create Hidden Layers
        self.hiddenLayerList=[]
        self.hiddenLayerList.append(
            FirstHiddenLayer(
                rng = P.rng,
                input = input,
                inputNum  = P.inputDimNum,
                outputNum = P.dnnWidth,
                dnnWidth  = P.dnnWidth,
                spliceWidth = P.spliceWidth,
                dropoutProb = P.dropoutHiddenProb,
                W = params[0],
                b = params[1],
                DROPOUT = DROPOUT ) )

        for i in xrange (P.dnnDepth - 1):
            self.hiddenLayerList.append(
                HiddenLayer(
                    input = self.hiddenLayerList[i].output,
                    rng = P.rng,
                    inputNum = P.dnnWidth/2,
                    outputNum = P.dnnWidth,
                    dnnWidth = P.dnnWidth,
                    dropoutProb = P.dropoutHiddenProb,
                    W = params[2 * (i + 1)],
                    b = params[2 * (i + 1) + 1],
                    DROPOUT = DROPOUT ) )

        # Output Layer
        self.outputLayer = OutputLayer(
              input = self.hiddenLayerList[P.dnnDepth - 1].output,
              inputNum = P.dnnWidth/2, 
              outputNum = P.outputPhoneNum,
              rng = P.rng,
              W = params[2 * P.dnnDepth],
              b = params[2 * P.dnnDepth + 1])
        
        # Weight decay
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = 0
        for i in xrange(P.dnnDepth):
             self.L1 += abs(self.hiddenLayerList[i].W).sum()
        self.L1 += abs(self.outputLayer.W).sum()
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(P.dnnDepth):
            self.L2_sqr += (self.hiddenLayerList[i].W ** 2).sum()
        self.L2_sqr += (self.outputLayer.W ** 2).sum()

        # CrossEntropy
        self.crossEntropy = ( self.outputLayer.crossEntropy )
        
        # Same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # Get the predict int for test set output
        self.yPred = self.outputLayer.y_pred
        
        # Get the probability
        self.p_y_given_x = self.outputLayer.p_y_given_x

        # Parameters of all DNN model
        self.params = self.hiddenLayerList[0].params
        for i in xrange(1, P.dnnDepth):
            self.params += self.hiddenLayerList[i].params
        self.params += self.outputLayer.params

        # keep track of model input
        self.input = input
