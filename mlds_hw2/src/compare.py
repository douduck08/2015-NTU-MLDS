import sys
from operator import itemgetter, attrgetter

f1Name = sys.argv[1]
f2Name = sys.argv[2]

f1 = open(f1Name, 'rb')
f2 = open(f2Name, 'rb')
# TODO debug the label error less 1
def pickLabel(f,flag):
    tmp = []
    for i in f:
        if i[0] == 'm' or i[0] == 'f':
            i = i.strip()
            i = i.split(',')
            if flag:
                tmp.append([i[0],str(int(i[1])-1)])
            else:
                tmp.append([i[0],i[1]])
    tmp = sorted(tmp, key=itemgetter(0))
    print tmp[5234]
    return tmp

data1 = pickLabel(f1, False)
data2 = pickLabel(f2, False)
same = [ i for i, j in zip(data1, data2) if i[1] == j[1]]
samePercent = float(len(same)*100) / float(len(data1))
print ("same number = %d" % (len(same)))
print ("percent = %f %%" % (100. - samePercent))
