import os
import sys
import utils
import dnn.dnn as dnn
import dnn.dnnUtils as dnnUtils
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
    P = dnnUtils.Parameters(setting)
    print P.outputFilename
    datasets  = utils.loadDataset(filename = P.datasetFilename, totalSetNum=3)

    if not USE_EXIST_MODEL: 
        sys.stdout = Logger(P.logFilename)
        bestModel = dnn.trainDNN(datasets, P)
        bestModelFilename = '../model/' + P.outputFilename + '.model'
        utils.makePkl(bestModel, P.bestModelFilename)
    else:
        # TODO use filename to build P
        bestModelFilename = sys.argv[2]
        bestModel = utils.loadPkl(bestModelFilename)
    
    dnn.getResult(bestModel, datasets[1], P, 'valid', P.validResultFilename)
    dnn.getResult(bestModel, datasets[2], P, 'test', P.testResultFilename)
    dnn.getProb(bestModel, datasets[0], P.trainProbFilename, P)
    dnn.getProb(bestModel, datasets[1], P.validProbFilename, P)
    dnn.getProb(bestModel, datasets[2], P.testProbFilename, P)

    smooth(noSmoothedFilename = P.testResultFilename, smoothedFilename = P.testSmoothedResultFilename)
    smooth(noSmoothedFilename = P.validResultFilename, smoothedFilename = P.validSmoothedResultFilename)
    tfit.transform(beforeTransformFilename = P.testSmoothedResultFilename, afterTransformFilename = '../result/final_result/' + P.outputFilename + '_smoothed.csv')
