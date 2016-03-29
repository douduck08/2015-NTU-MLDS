import os
import sys
import utils
import rnn.rnn as rnn
import rnn.rnnUtils as rnnUtils
import postprocessing as pp 
import transformIntToLabel as tfit
setting = sys.argv[1]
USE_EXIST_MODEL = False

def smooth(noSmoothedFilename, smoothedFilename):
    name, label = utils.readFile(noSmoothedFilename)
    endIndxGroup = pp.findEndIndxofGroup(name = name, label = label)
    label = pp.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
    pp.writeFile(filename = smoothedFilename, name = name, label = label)

class Logger(object):
    def __init__(self, logFilename):
        self.terminal = sys.stdout
        self.log = open(logFilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

if __name__ == '__main__':
    P = rnnUtils.Parameters(setting)
    print P.outputFilename
    datasets  = utils.loadDataset(filename = P.datasetFilename, totalSetNum=3)
   
    # Redirect stdout to log file
    sys.stdout = Logger(P.logFilename)

    # train RNN model
    bestModelFilename = rnn.trainRNN(datasets, P)
    
    # Get result
    rnn.getResult(bestModelFilename, datasets)

    # Smooth
    smooth(noSmoothedFilename = P.testResultFilename, smoothedFilename = P.testSmoothedResultFilename)
    smooth(noSmoothedFilename = P.validResultFilename, smoothedFilename = P.validSmoothedResultFilename)
"""
    tfit.transform(beforeTransformFilename = P.testSmoothedResultFilename, afterTransformFilename = '../result/final_result/' + P.outputFilename + '_smoothed.csv')
"""
