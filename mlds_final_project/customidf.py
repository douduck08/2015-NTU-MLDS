import csv
import math
import numpy as np
import resource
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', type=str, default='custom')
    parser.add_argument('-memory_limit', type=float, default=6.0)
    return parser.parse_args()

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

customLevel = 300
customList = ['how', 'what', 'why', 'where', 'who', 'which', 'many', 'color', 'number', 'time']
removeList = ['?', '\'s', ',', '(', ')', ':', '\\', '\'']
def splitQuestion(question, remove_duplicate = True):
    for pattern in removeList:
        question = question.replace(pattern,'')
    if remove_duplicate:
        return list(set(question.split(' ')))
    else:
        return question.split(' ')

if __name__ == "__main__":
    arg = parseArgs()
    limit_memory(arg.memory_limit * 1e9)  # default about 6GB

    print '*** count the words ***'
    questionCount = 0
    wordMap = {}
    with open('data/processed_text/question_processed.train', 'r') as questionfile:
        reader = csv.reader(questionfile, delimiter = '\t')
        for question in reader:
            if question[0] == 'img_id':
                continue
            questionCount += 1
            words = splitQuestion(question[2])
            for word in words:
                if wordMap.has_key(word):
                    wordMap[word] += 1
                else:
                    wordMap[word] = 1
    with open('data/processed_text/question_processed.test', 'r') as questionfile:
        reader = csv.reader(questionfile, delimiter = '\t')
        for question in reader:
            if question[0] == 'img_id':
                continue
            questionCount += 1
            words = splitQuestion(question[2])
            for word in words:
                if wordMap.has_key(word):
                    wordMap[word] += 1
                else:
                    wordMap[word] = 1

    print '*** custom the important words ***'
    if arg.type == 'custom':
        print '    custom mode'
        dictFilename = 'custom_idf_dic.csv'
        filename = 'question_custom_idf'
        for word in customList:
            wordMap[word] /= customLevel
    else:
        print '    standard mode'
        dictFilename = 'standard_idf_dic.csv'
        filename = 'question_standard_idf'

    print '*** calculate the idf of words ***'
    words = wordMap.keys()
    with open('data/idf_weight/' + dictFilename, 'w') as outfile:
        idf = csv.writer(outfile)
        idf.writerow(['word', 'idf'])
        for word in words:
            wordMap[word] = math.log(questionCount / wordMap[word])
            idf.writerow([word, wordMap[word]])

    print '*** build the question weight file ***'
    with open('data/processed_text/question_processed.train', 'r') as questionfile:
        reader = csv.reader(questionfile, delimiter = '\t')
        with open('data/idf_weight/' + filename + '.train', 'w') as outfile:
            writer = csv.writer(outfile, delimiter = ' ')
            idx = 0
            for question in reader:
                if question[0] == 'img_id':
                    continue
                idx += 1
                if idx % 10000 == 0:
                    print 'qeustion #{:d}'.format(idx)
                words = splitQuestion(question[2], remove_duplicate = False)
                weights = np.zeros(len(words), dtype = 'float32')
                for i in xrange(len(words)):
                    weights[i] = wordMap[words[i]]
                writer.writerow([question[1]] + weights.tolist())
    with open('data/processed_text/question_processed.test', 'r') as questionfile:
        reader = csv.reader(questionfile, delimiter = '\t')
        with open('data/idf_weight/' + filename + '.test', 'w') as outfile:
            writer = csv.writer(outfile, delimiter = ' ')
            idx = 0
            for question in reader:
                if question[0] == 'img_id':
                    continue
                idx += 1
                if idx % 10000 == 0:
                    print 'qeustion #{:d}'.format(idx)
                words = splitQuestion(question[2], remove_duplicate = False)
                weights = np.zeros(len(words), dtype = 'float32')
                for i in xrange(len(words)):
                    weights[i] = wordMap[words[i]]
                writer.writerow([question[1]] + weights.tolist())




