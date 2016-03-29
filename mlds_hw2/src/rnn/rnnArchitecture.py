import numpy
import theano
import theano.tensor as T

init = 1.
bias = 0.1
STD = 0.1
CutSize = 100
BatchSize = 10

def sigmoid(z, alpha):
    return ( 1/(1+T.exp((-z) )) ).astype(dtype=theano.config.floatX)

def softmax(z):
    maxZ = T.max(z, axis=1).astype(dtype = theano.config.floatX)
    absMaxZ = T.abs_(maxZ)
    absMaxZ = T.reshape(absMaxZ, (absMaxZ.shape[0], 1))
    expZ = T.exp( z*10  / absMaxZ)
    expZsum = T.sum(expZ, axis=1).astype(dtype = theano.config.floatX)
    expZsum = T.reshape(expZsum, (expZsum.shape[0],1))
    return expZ /expZsum

class HiddenLayer(object):
    def __init__(self, rng, input, inputNum, outputNum, W_i1 = None, W_h1 = None, b_h1 = None, W_i2 = None, W_h2 = None, b_h2 = None):

        # For in order input
        if W_i1 is None:
            W_i1Values = rng.normal( loc = 0.0, scale = STD, size = (inputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_i1 = theano.shared(value = W_i1Values, name = 'W', borrow = True)
        else:
            W_i1 = theano.shared( value = numpy.array(W_i1, dtype = theano.config.floatX), name='W', borrow = True )

        if W_h1 is None:
            W_h1Values = rng.normal( loc = 0.0, scale = STD, size = (outputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_h1 = theano.shared(value = W_h1Values, name = 'W', borrow = True)
        else:
            W_h1 = theano.shared( value = numpy.array(W_h1, dtype = theano.config.floatX), name='W', borrow = True )

        if b_h1 is None:
            b_h1Values = rng.normal( loc = 0.0, scale = 0.1, size = (outputNum, ) ).astype( dtype=theano.config.floatX )
            b_h1 = theano.shared(value = b_h1Values, name = 'b', borrow = True)
        else:
            b_h1 = theano.shared( value = numpy.array(b_h1, dtype = theano.config.floatX), name='b', borrow = True )

        # For in reverse input
        if W_i2 is None:
            W_i2Values = rng.normal( loc = 0.0, scale = STD, size = (inputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_i2 = theano.shared(value = W_i2Values, name = 'W', borrow = True)
        else:
            W_i2 = theano.shared( value = numpy.array(W_i2, dtype = theano.config.floatX), name='W', borrow = True )

        if W_h2 is None:
            W_h2Values = rng.normal( loc = 0.0, scale = STD, size = (outputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_h2 = theano.shared(value = W_h2Values, name = 'W', borrow = True)
        else:
            W_h2 = theano.shared( value = numpy.array(W_h2, dtype = theano.config.floatX), name='W', borrow = True )

        if b_h2 is None:
            b_h2Values = rng.normal( loc = 0.0, scale = 0.1, size = (outputNum, ) ).astype( dtype=theano.config.floatX )
            b_h2 = theano.shared(value = b_h2Values, name = 'b', borrow = True)
        else:
            b_h2 = theano.shared( value = numpy.array(b_h2, dtype = theano.config.floatX), name='b', borrow = True )

        self.W_i1 = W_i1
        self.W_h1 = W_h1
        self.b_h1 = b_h1
        self.W_i2 = W_i2
        self.W_h2 = W_h2
        self.b_h2 = b_h2
        self.output=[]

        # For sigmoid alpha parameter
        # Remember to add alpha to parameter list if you turn it on
        """ self.alp = theano.shared(value=0.5)  """

        # Output_info for scan
        a_0 = theano.shared(numpy.zeros(outputNum).astype(dtype = theano.config.floatX), borrow = True)

        # In order
        def inOrderStep(z_t, a_tm1):
            return sigmoid( (z_t + T.dot(a_tm1, self.W_h1) + self.b_h1 ), 1.0)

        self.z_seq = sigmoid(T.dot(input[0], W_i1), 1.0)
        a_seq, _ = theano.scan(inOrderStep, sequences = self.z_seq, outputs_info = a_0, truncate_gradient = -1)
        self.output.append(a_seq)

        # In reverse
        def inReverseStep(z_t, a_tm1):
            return sigmoid( (z_t + T.dot(a_tm1, self.W_h2) + self.b_h2 ), 1.0)

        z_seq_reverse = T.dot(input[1], W_i2)
        a_seq_reverse, _ = theano.scan(inReverseStep, sequences = z_seq_reverse, outputs_info = a_0, truncate_gradient = -1)
        self.output.append(a_seq_reverse)

        # Save parameters
        self.params = [self.W_i1, self.W_h1, self.b_h1, self.W_i2, self.W_h2, self.b_h2]

class OutputLayer(object):
    def __init__(self, input, inputNum, outputNum, rng, W_o = None, b_o = None):
        if W_o is None:
            W_values = rng.normal( loc = 0.0, scale = STD, size = (inputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_o = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W_o = theano.shared( value = numpy.array(W_o, dtype = theano.config.floatX), name='W', borrow=True )

        if b_o is None:
            b_values = rng.normal( loc = 0.0, scale = 0.1, size = (outputNum, ) ).astype( dtype=theano.config.floatX )
            b_o = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_o = theano.shared( value = numpy.array(b_o, dtype = theano.config.floatX), name='b', borrow=True )

        self.W_o = W_o
        self.b_o = b_o

        # get average of in order input and in reverse input
        averageInput = (input[0] + input[1]) / 2

        y_seq = softmax( T.dot(averageInput, self.W_o) + b_o )

        # Find probability, given x
        self.p_y_given_x = y_seq

        # Find largest y_i
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Save parameters
        self.params = [self.W_o, self.b_o]

    # Cross entropy
    def crossEntropy(self, y):
        tmp = T.log(self.p_y_given_x)
        sumAll = 0
        for i in xrange(CutSize):
            for j in xrange(BatchSize):
                if y[j][i] != -1:
                    sumAll += tmp[i][j][ y[j][i] ]
        return sumAll / BatchSize
        # return -T.sum( T.log(self.p_y_given_x)[T.arange(y.shape[0]), y] )

    def errors(self, y):
        # Check y and y_pred dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError( 'y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type) )
        # Check if y is the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class RNN(object):
    def __init__(self, input, P, params = None, DROPOUT = False):

        self.hiddenLayerList = []
        bidirectionalInput = []
        bidirectionalInput.append(input)
        bidirectionalInput.append(input[::-1])

        # First hidden layer
        self.hiddenLayerList.append(
            HiddenLayer( input = bidirectionalInput, rng = P.rng, inputNum = P.inputDimNum, outputNum = P.rnnWidth,
                         W_i1 = params[0], W_h1 = params[1], b_h1 = params[2],
                         W_i2 = params[3], W_h2 = params[4], b_h2 = params[5] ))

        # Other hidden layers
        for i in xrange (P.rnnDepth - 1):
            self.hiddenLayerList.append(
                HiddenLayer( input = self.hiddenLayerList[i].output, rng = P.rng, inputNum = P.rnnWidth, outputNum = P.rnnWidth,
                             W_i1 = params[6 * (i + 1)], W_h1 = params[6 * (i + 1) + 1], b_h1 = params[6 * (i + 1) + 2],
                             W_i2 = params[6 * (i + 1) + 3], W_h2 = params[6 * (i + 1) + 4], b_h2 = params[6 * (i + 1) + 5] ))
        # Output Layer
        self.outputLayer = OutputLayer( input = self.hiddenLayerList[P.rnnDepth - 1].output, rng = P.rng, inputNum = P.rnnWidth,
                                        outputNum = P.outputPhoneNum, W_o = params[6 * P.rnnDepth], b_o = params[6 * P.rnnDepth+1] )

        # Weight decay
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = 0
        for i in xrange(P.rnnDepth):
             self.L1 = ( self.L1 + abs(self.hiddenLayerList[i].W_i1).sum() + abs(self.hiddenLayerList[i].W_h1).sum()
                                 + abs(self.hiddenLayerList[i].W_i2).sum() + abs(self.hiddenLayerList[i].W_h2).sum() )
        self.L1 += abs(self.outputLayer.W_o).sum()

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(P.rnnDepth):
            self.L2_sqr = ( self.L2_sqr + (self.hiddenLayerList[i].W_i1 ** 2).sum() + (self.hiddenLayerList[i].W_h1 ** 2).sum()
                                        + (self.hiddenLayerList[i].W_i2 ** 2).sum() + (self.hiddenLayerList[i].W_h2 ** 2).sum() )
        self.L2_sqr += (self.outputLayer.W_o ** 2).sum()

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
        for i in xrange(1, P.rnnDepth):
            self.params += self.hiddenLayerList[i].params
        self.params += self.outputLayer.params

        # For sigmoid alpha parameter
        # Remember to add alpha to parameter list if you turn it on
        """self.params += [self.hiddenLayerList[0].alp]"""
