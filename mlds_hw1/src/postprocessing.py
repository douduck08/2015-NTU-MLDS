from utils import namepick, readFile
import sys
#preLabelFilename = sys.argv[1]
#correctLabelFilename = sys.argv[2] 


def findEndIndxofGroup(name, label):
    curName = namepick(name[0])
    endIndxGroup = []
    for i in xrange(len(name)):
        if(curName != namepick(name[i])):
            endIndxGroup.append(i)
    endIndxGroup.append(len(name))
    return endIndxGroup 

def correctLabel(endIndxGroup, name, label):
#    indxOfGroup = 0
#    curEnd = endIndxGroup[0]
#    firstLabelOfcurGroup = True
#    for i in xrange(1, len(name)):
#        if firstLabelOfcurGroup:
#            firstLabelOfcurGroup = False
#            continue
#        else:
#            if i == ( endIndxGroup[indxOfGroup] - 1 ):
#                indxOfGroup += 1
#                firstLabelOfcurGroup = True
#                continue
#            if( label[i - 1] == label[i + 1] and label[i - 1] != label[i] ):
#                label[i] = label[i - 1]
#        if i == len(name) - 2:
#            break
    MAX_ALLOW       = 2
    currentFlag     = label[0]
    currentOdd      = label[len(name)/2]
    currentContinue = 0
    hitNumber       = 0
    for i in xrange(len(name)):
      if i == 0:
        currentContinue += 1
      elif i == (len(name) - 1): 
        currentContinue += 1
        for j in xrange(currentContinue):
          label[i-j] = currentFlag
      else:
        if label[i] == currentFlag:
          currentContinue += 1
        elif label[i] == currentOdd:
          hitNumber += 1
          currentContinue += 1
          if hitNumber == MAX_ALLOW:
            for j in xrange(currentContinue-MAX_ALLOW):
              label[i-j-MAX_ALLOW] = currentFlag
            currentContinue = MAX_ALLOW
            hitNumber   = 0
            currentOdd  = label[i-1]
            currentFlag = label[i]
        else:
          currentOdd      = label[i]
          hitNumber       = 1
          currentContinue += 1
        
        
    return label

def writeFile(filename, name, label):
    f = open(filename, 'w')
    for i in xrange(len(name)):
        f.write(name[i] + ',' + label[i])
    f.close()
if __name__ == '__main__':
    name, label = readFile(preLabelFilename)
    endIndxGroup = findEndIndxofGroup(name = name, label = label)
    label = correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
    writeFile(filename = correctLabelFilename, name = name, label = label)
