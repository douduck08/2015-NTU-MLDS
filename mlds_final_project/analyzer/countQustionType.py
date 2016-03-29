import os
import sys
import csv
import numpy as np
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='error')
    parser.add_argument('-dir', type=str, default='./error')
    return parser.parse_args()

if __name__ == "__main__":
    arg = parseArgs()

    if arg.mode == 'error':
        print '*** error mode ***'
        fileList = []
        for file in os.listdir(arg.dir):
            if file.endswith('.csv'):
                fileList.append(file)
        typeMap = {}
        with open('annotation.train', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            for row in reader:
                if row[0] != 'img_id':
                    typeMap[int(row[1])] = row[3].replace('question_type:"','').replace('"','')
        typeCount = {}
        for filename in fileList:
            with open('./error/' + filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] != 'q_id':
                        if typeCount.has_key(typeMap[int(row[0])]):
                            typeCount[ typeMap[int(row[0])] ] += int(row[1])
                        else:
                            typeCount[ typeMap[int(row[0])] ] = int(row[1])
        with open('error_question_type_count.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['question_type', 'count'])
            for item in typeCount.items():
                writer.writerow([item[0], item[1]])

    else:
        print '*** all question mode ***'
        typeMap = {}
        typeCount = {}
        with open('annotation.train', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            for row in reader:
                if row[0] != 'img_id':
                    typeMap[int(row[1])] = row[3].replace('question_type:"','').replace('"','')
                    if typeCount.has_key(typeMap[int(row[1])]):
                        typeCount[ typeMap[int(row[1])] ] += 1
                    else:
                        typeCount[ typeMap[int(row[1])] ] = 1
        with open('all_question_type_count.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['question_type', 'count'])
            for item in typeCount.items():
                writer.writerow([item[0], item[1]])
