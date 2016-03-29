import re
import csv
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', type=str, default='train')
    return parser.parse_args()

def needPick(str):
    patterns = ['^how many', '^what number', '^what time']
    for pattern in patterns:
        if re.search(pattern, str):
            return True
    return False

if __name__ == "__main__":
    arg = parseArgs()

    typeMap = {}
    typeCount = {}
    with open('analyzer/annotation.train', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            if row[0] != 'img_id':
                typeMap[int(row[1])] = row[3].replace('question_type:"','').replace('"','')

    if arg.set == 'train':
        print '*** train data set ***'
        number = csv.writer(open('data/preprocessed/id_train_number.csv', 'w'), delimiter = ' ')
        not_number = csv.writer(open('data/preprocessed/id_train_notnumber.csv', 'w'), delimiter = ' ')
        questionfile = open('data/processed_text/question_processed.train', 'r')
        reader = csv.reader(questionfile, delimiter = '\t')
        for row in reader:
            if row[0] == 'img_id':
                continue
            if needPick(row[2]):
                number.writerow([row[0], row[1]])
                if typeCount.has_key(typeMap[int(row[1])]):
                    typeCount[ typeMap[int(row[1])] ] += 1
                else:
                    typeCount[ typeMap[int(row[1])] ] = 1
            else:
                not_number.writerow([row[0], row[1]])
        print typeCount

    elif arg.set == 'test':
        print '*** test data set ***'
        number = csv.writer(open('data/preprocessed/id_test_number.csv', 'w'), delimiter = ' ')
        not_number = csv.writer(open('data/preprocessed/id_test_notnumber.csv', 'w'), delimiter = ' ')
        questionfile = open('data/processed_text/question_processed.test', 'r')
        reader = csv.reader(questionfile, delimiter = '\t')
        for row in reader:
            if row[0] == 'img_id':
                continue
            if needPick(row[2]):
                number.writerow([row[0], row[1]])
            else:
                not_number.writerow([row[0], row[1]])
    else:
        print '*** error data set ***'
