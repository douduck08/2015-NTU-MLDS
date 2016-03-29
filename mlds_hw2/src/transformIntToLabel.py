from utils import readFile
import sys

def transform(beforeTransformFilename, afterTransformFilename):
    phone_48_int_map = '../map/phone_48_int.map'
    phone_48_39_int_map = '../map/phone_48_39_int.map'
    f1 = open(phone_48_int_map, 'r')
    f2 = open(phone_48_39_int_map, 'r')
    f3 = open(afterTransformFilename, 'w')

    d_int_48_map = {}
    for i in f1:
        i = i.strip()
        i = i.strip('\n')
        i = i.split()
        d_int_48_map[i[1]] = i[0]

    name, label = readFile(beforeTransformFilename)
    newlabel = [''] * len(label)
    for i in xrange(len(name)):
        newlabel[i] = d_int_48_map[label[i].strip('\n')]

    d_48_39_map = {}
    for i in f2:
        i = i.strip()
        i = i.split()
        d_48_39_map [i[0]] = i[1]

    for i in xrange(len(newlabel)):
        newlabel[i] = d_48_39_map[newlabel[i]]

    f3.write('Id,Prediction\n')
    for i in xrange(len(name)):
        f3.write(name[i] + ',' + newlabel[i] + '\n')

