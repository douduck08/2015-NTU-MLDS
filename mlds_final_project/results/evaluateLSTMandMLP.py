import sys
import resource
import csv
import string
import numpy as np
import argparse
from os.path import basename
from keras.models import model_from_json
# from spacy.en import English
img_dim = 4096
word_vec_dim = 300

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-language_feature_dim', type=int, default=300)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-question_feature', type=str, required=True)
    parser.add_argument('-choice_feature', type=str, required=True)
    return parser.parse_args()

def getImageFeature(imageData, idList):
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
    featureData = {}
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            featureData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')
    return featureData

def loadFeatureData(fileName):
    featureData = {}
    with open(fileName, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            featureData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')
    return featureData

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
    if arg.predict_type == 'test':
        imageData = loadFeatureData(fileName = './data/image_feature/caffenet_4096_test.csv')
    elif arg.predict_type == 'train':
        answerData = loadAnswerData()
        imageData = loadFeatureData(fileName = './data/image_feature/caffenet_4096_train.csv')
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
            y_predict.extend(model.predict(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                 getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                           verbose = 0))
        elif arg.language_feature_dim == 300:
            y_predict.extend(model.predict(X = [ getImageFeature(imageData, imageIdListForBatch),
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
        print '*** choose answer ***'
        results_file = './results/' + basename(arg.weights).replace('.hdf5', '_result.txt')
        with open(results_file, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['q_id', 'ans'])
            for i in xrange(len(idList)):
                writer.writerow([idList[i], answers_predict[i]])
    elif arg.predict_type == 'train':
        print '*** calculate error ***'
        error = 0
        for i in xrange(len(idList)):
            if answers_predict[i] != answerData[ idList[i] ]:
                error += 1
        print 'About modle: ' + arg.weights
        print 'Error = {:.03f}'.format(1.0 * error / len(idList))

