from utils import readFile
import sys
import os

inputDirectoryName = sys.argv[1]
outputFileName = sys.argv[2]
# if the directory is in Current Working Directory (i.e. CWD)
# directly type


def constructList(targetDirectory):
  
  labelList = []
  numOfList = 0
  nameList = []
  for filename in os.listdir( os.getcwd() + '/' + targetDirectory ):
    numOfList += 1 #trivial
    dummy_name, dummy_label = readFile(os.getcwd() + '/' + targetDirectory +filename) #read in the file
    labelList.append(dummy_label)
    nameList = dummy_name

  return labelList, numOfList, nameList

def vote(labelList, numOfList, nameList):
  candidate = []
  results = []
  for i in range(48):
      candidate.append(i)

  for i in range(len(nameList)):
      candidateCount = []
      for j in range(48):
        candidateCount.append(0)
        # initialize the values in candidate count
      for j in range(numOfList):
        dummy_index = candidate.index(int(labelList[j][i].strip('\n')))
        candidateCount[dummy_index] += 1

      results.append(candidateCount.index(max(candidateCount)))

  return results

def writeFile(filename, name, label):
    f = open(filename, 'w')
    for i in xrange(len(name)):
        f.write(name[i] + ',' + str(label[i])+'\n')
    f.close()

if __name__ == '__main__':
  labelList, numOfList, name = constructList(inputDirectoryName)
  results = vote(labelList, numOfList, name)
  writeFile(filename = outputFileName, name = name, label = results)
