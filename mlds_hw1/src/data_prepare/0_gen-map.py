import sys
import operator

dataPath = sys.argv[1]
state_48_39_ori_map = dataPath + '/phones/state_48_39.map'
phone_48_39_ori_map = dataPath + '/phones/48_39.map'
state_label_map     = '../../map/state_label.map'
phone_48_39_int_map = '../../map/phone_48_39_int.map'
phone_48_int_map    = '../../map/phone_48_int.map'

f1 = open(state_48_39_ori_map, 'r')
f2 = open(phone_48_39_ori_map, 'r')
f3 = open(state_label_map, 'w')
f4 = open(phone_48_39_int_map, 'w')
f6 = open(phone_48_int_map, 'w')
d_s_48 = {}
d_s_48_39 = {}
d_48_39 = {}

d_39_int = {}
# phone label map: 39 to int
ii = 0
for i in f2:
    i = i.strip()
    i = i.split()
    if i[1] not in d_39_int:
        d_39_int[i[1]] = ii
        ii += 1

f2.seek(0)

# phone label: 48 to 39
for i in f2:
    i = i.strip()
    i = i.split()
    tmp = i[0] + '\t' + i[1] + '\t' + str(d_39_int[i[1]]) + '\n'
    f4.write(tmp) 

f2.seek(0)

ii = 0
for i in f2:
    i = i.strip()
    i = i.split()
    tmp = i[0] + '\t' + str(ii) + '\n'
    ii += 1
    f6.write(tmp) 
