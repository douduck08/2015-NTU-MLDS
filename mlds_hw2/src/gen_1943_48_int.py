import os
import sys
import subprocess
import glob
from os import path

map_3   = sys.argv[1]
map_2   = sys.argv[2]
outFile = sys.argv[3]

def readFile2_column(filename):
    f = open(filename, 'r')
    label_48 = []
    for i in f:
        part = i.split('\t')
        label_48.append(part[0])
    f.close()
    return label_48

def readFile3_column(filename):
    f = open(filename, 'r')
    label_1943 = []
    label_48   = []
    label_39   = []
    for i in f:
        part = i.split('\t')
        label_1943.append(part[0])
        label_48.append(part[1])
    f.close()
    return label_1943, label_48

if __name__ == '__main__':

    label_1943, label_48 = readFile3_column(map_3)
    col_2_label_48       = readFile2_column(map_2)

    f = open(outFile, 'w')
    sys.stdout = f

    for i in xrange(len(label_1943)):
        print str(label_1943[i]) + ',' + str(col_2_label_48.index(str(label_48[i])))

    f.close()
