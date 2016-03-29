import sys
import resource
import csv
import gzip
import cPickle as pickle
import string
import numpy as np
import argparse
from os.path import basename, exists
from keras.models import model_from_json
# from spacy.en import English
#img_dim = 4096
word_vec_dim = 300

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-idim', '--image_feature_dim', type=int, required=True)
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-ldim', '--language_feature_dim', type=int, required=True)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-w', '--weights', type=str, required=True)
    parser.add_argument('-if', '--image_feature', type=str, required=True)
    parser.add_argument('-qf', '--question_feature', type=str, required=True)
    parser.add_argument('-cf', '--choice_feature', type=str, required=True)
    parser.add_argument('-print_error_id', type=bool, default=False)
    parser.add_argument('-use_error_file', type=bool, default=True)
    return parser.parse_args()

def getImageFeature(imageData, idList, img_dim):
    batchSize = len(idList)
    imageMatrix = np.zeros((batchSize, img_dim), dtype = 'float32')
    for i in xrange(batchSize):
        imageMatrix[i,:] = imageData[ idList[i] ]
    return imageMatrix

def getQuestionFeature(questionData, idList):
    batchSize = len(idList)
    featureMatrix = np.zeros((batchSize, word_vec_dim), dtype = 'float32')
    for i in xrange(batchSize):
        featureMatrix[i,:] = questionData[idList[i]]
    return featureMatrix

def getLanguageFeature(questionData, choiceData, idList):
    batchSize = len(idList)
    featureMatrix = np.zeros((batchSize, word_vec_dim * 6), dtype = 'float32')
    for i in xrange(batchSize):
        featureMatrix[i,:] = np.hstack((questionData[idList[i]], choiceData[idList[i]]))
    return featureMatrix

def getAnswerFeature(choiceData, answerData, idList):
    batchSize = len(idList)
    answerMatrix = np.zeros((batchSize, word_vec_dim), dtype = 'float32')
    for i in xrange(batchSize):
        answerMatrix[i,:] = choiceData[idList[i]][ answerData[idList[i]]*word_vec_dim : (answerData[idList[i]]+1)*word_vec_dim ]
    return answerMatrix

def loadIdMap(predict_type):
    idMap = {}
    with open('./data/preprocessed/id_' + predict_type + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])
    return idMap

def loadAnswerData():
    answerData = {}
    with open('./data/preprocessed/id_answer_label_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            answerData[int(row[0])] = row[1]
    return answerData

def loadFeatureData(fileName):
    f = gzip.open(fileName, 'rb')
    return pickle.load(f)
"""
def loadFeatureData(fileName):
    featureData = {}
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            featureData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')
    return featureData
"""
def prepareIdList(idList, batchSize):
    questionNum = len(idList)
    batchNum = questionNum / batchSize
    # random.shuffle(idList)
    idListInBatch = []
    for i in xrange(batchNum):
        idListInBatch.append( idList[i * batchSize : (i+1) * batchSize] )
    if questionNum % batchSize != 0:
        idListInBatch.append( idList[batchNum * batchSize :] )
        batchNum += 1
    return idListInBatch, batchNum

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def cos_sim(y_true, y_pred):
    dot = np.sum(y_true * y_pred)
    u = np.sqrt(np.sum(np.square(y_true)))
    v = np.sqrt(np.sum(np.square(y_pred)))
    return 1 - dot / (u * v + 0.0001)

def loadErrorMap(usethefile = True):
    errorMap = {}
    if usethefile:
        if exists('analyzer/error/all_errorid.csv'):
             with open('analyzer/error/all_errorid.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] != 'q_id':
                        errorMap[ int(row[0]) ] = int(row[1])
    return errorMap

def writeErrorIdMap(errorMap, filename):
    with open(filename, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['q_id', 'error_times'])
        for item in errorMap.items():
            writer.writerow([item[0], item[1]])

if __name__ == "__main__":
    arg = parseArgs()
    # nlp = English()
    if arg.predict_type == 'test':
        print '*** predict type: test ***'
    elif arg.predict_type == 'train':
        print '*** predict type: train ***'
    else:
        raise Exception("predict type error!")

    print '*** load model ***'
    model = model_from_json( open(arg.model).read() )
    model.load_weights(arg.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print '*** load data ***'
    imageData = loadFeatureData(fileName = arg.image_feature)
    if arg.predict_type == 'train':
        answerData = loadAnswerData()

    idMap = loadIdMap(arg.predict_type)
    questionData = loadFeatureData(fileName = arg.question_feature)
    choiceData = loadFeatureData(fileName = arg.choice_feature)

    print '*** predict ***'
    y_predict = []
    batchSize = 512
    idList = idMap.keys()
    questionIdList, batchNum = prepareIdList(idList, batchSize)
    for j in xrange(batchNum):
        imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
        if arg.language_feature_dim == 1800:
            y_predict.extend(model.predict(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                 getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                           verbose = 0))
        elif arg.language_feature_dim == 300:
            y_predict.extend(model.predict(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                 getQuestionFeature(questionData, questionIdList[j]) ],
                                           verbose = 0))
        else:
            raise Exception("language feature dim error!")
    print 'y.shape = ' + str(y_predict[0].shape)
    # print y_predict[0]

    print '*** choose answer ***'
    label = ['A', 'B', 'C', 'D', 'E']
    answers_predict = []
    for i in xrange(len(idList)):
        loss = 10
        answer = -1
        for j in xrange(5):
            current_id = idList[i]
            current_loss = cos_sim(y_predict[i], choiceData[ idList[i] ][ j*300:(j+1)*300 ])
            if current_loss < loss:
                answer = j
                loss = current_loss
        answers_predict.append(label[answer])

    if arg.predict_type == 'test':
        print '*** print answer ***'
        results_file = './results/' + basename(arg.weights).replace('.hdf5', '_result.txt')
        with open(results_file, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['q_id', 'ans'])
            for i in xrange(len(idList)):
                writer.writerow([idList[i], answers_predict[i]])
    elif arg.predict_type == 'train':
        print '*** calculate error ***'
        error = 0
        errorMap = loadErrorMap(arg.use_error_file)
        for i in xrange(len(idList)):
            if answers_predict[i] != answerData[ idList[i] ]:
                error += 1
                if errorMap.has_key(idList[i]):
                    errorMap[ idList[i] ] += 1
                else:
                    errorMap[ idList[i] ] = 1
        if arg.print_error_id == True:
            if arg.use_error_file == True:
                writeErrorIdMap(errorMap, 'analyzer/error/all_errorid.csv')
            else:
                writeErrorIdMap(errorMap, 'analyzer/error/' + basename(arg.weights).replace('.hdf5', '_errorid.csv'))
        print 'About modle: ' + arg.weights
        print 'Error = {:.03f}'.format(1.0 * error / len(idList))
