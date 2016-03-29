import dnn

modelList = ''


def genWidthList(D, Ws):
    WsNum = len(Ws)
    Wlist = []
    if WsNum != D:
        Wlist = [Ws[0]] * ( D - WsNum + 1)
    for i in xrange(WsNum - 1):
        Wlist.append(Ws[i+1])
    return Wlist

def genVote():
    modelFilename = []
    f1 = open('modelList', 'rb')
    for i in xrange()
        modelFilename.append(f1.readline)
    
    for i in xrange(3):
        parentModelFileName = modelFilename[i]
        model = readModel(parentModelFileName)
        for j in xrange(1:4):
          childModelFileName = parentModelFileName + '-' + j
          childP = model[0]
          childParams = model[1]
          if i == 1:
              childP.dnnWidth[childDeep] *= 2
          elif i == 3:
              childP.dnnWidth[childDeep] /= 2
          childParams[childDeep*2] = None
          childParams[childDeep*2+1] = None
          childModel = [childP, childParams]
          writeModel(childModelFileName, childModel)

          childModelFileName = parentModelFileName + '-' + (j+3)
          childP = model[0]
          childParams = model[1]
          if i == 1:
              childP.dnnWidth[childDeep].append(childP.dnnWidth[childDeep] *= 2)
          elif i == 1:
              childP.dnnWidth[childDeep].append(childP.dnnWidth[childDeep])
          elif i == 3:
              childP.dnnWidth[childDeep].append(childP.dnnWidth[childDeep] /= 2)
          childParams.append(None)
          childParams.append(None)
          childModel = [childP, childParams]
          writeModel(childModelFileName, childModel)

def readModel(fileName):
    # read model in file, return a list: [P, params]

def writeModel(fileName, model)
    # write model into a file
