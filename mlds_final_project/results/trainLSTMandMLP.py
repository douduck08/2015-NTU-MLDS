import os
import resource
import csv
import random
import numpy as np
import theano.tensor as T
import argparse

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape, Merge, Dense, MaxoutDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

# from spacy.en import English
img_dim = 4096
word_vec_dim = 300

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i_dim', '--image_feature_dim', type=int, default=4096)
    parser.add_argument('-l_dim', '--language_feature_dim', type=int, default=300)
    parser.add_argument('-question_feature', type=str, required=True)
    parser.add_argument('-choice_feature', type=str, required=True)
    # lstm setting
    parser.add_argument('-lstm', type=bool, default=False)
    parser.add_argument('-lstm_units', type=int, default=512)
    parser.add_argument('-lstm_layers', type=int, default=1)
    # mlp setting
    parser.add_argument('-u', '--mlp_units', nargs='+', type=int, required=True)
    parser.add_argument('-a', '--mlp_activation', type=str, default='softplus')
    parser.add_argument('-odim', '--mlp_output_dim', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=1.0)
    parser.add_argument('-maxout', type=bool, default=False)
    # train setting
    parser.add_argument('-memory_limit', type=float, default=6.0)
    parser.add_argument('-cross_valid', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-epochs', type=int, default=150)
    # parser.add_argument('-lr', type=float, default=0.1)
    # parser.add_argument('-momentum', type=float, default=0.9)
    return parser.parse_args()

# def getQuestionWordVector(questionData, idList, wordVectorModel):
#     batchSize = len(idList)
#     maxlen = 0
#     tokens = []
#     for i in xrange(batchSize):
#         tokens.append( wordVectorModel( questionData[ idList[i] ].decode('utf8')) )
#         maxlen = max(maxlen, len(tokens[i]))
#     questionMatrix = np.zeros((batchSize, maxlen, word_vec_dim), dtype = 'float32')
#     for i in xrange(batchSize):
#         for j in xrange(len(tokens[i])):
#             questionMatrix[i,j,:] = tokens[i][j].vector
#     return questionMatrix

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

def loadIdMap():
    idMap = {}
    with open('./data/preprocessed/id_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])
    return idMap

def loadAnswerData():
    answerData = {}
    with open('./data/preprocessed/id_answer_category_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            answerData[int(row[0])] = int(row[1])
    return answerData

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
    random.shuffle(idList)
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
    dot = T.sum(y_true * y_pred, axis = 1)
    u = T.sqrt(T.sum(T.sqr(y_true), axis = 1))
    v = T.sqrt(T.sum(T.sqr(y_pred), axis = 1))
    return 1 - dot / (u * v + 0.0001)

if __name__ == '__main__':
    arg = parseArgs()
    # limit_memory(arg.memory_limit * 1e9)  # about 6GB
    max_len = 30
    # wordVectorModel = English()

    # build model
    image_model = Sequential()
    image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))

    language_model = Sequential()
    if arg.lstm is True:
        if arg.lstm_layers == 1:
            language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
        else:
            language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
            for i in xrange(arg.lstm_layers - 2):
                language_model,add(LSTM(output_dim = arg.lstm_units, return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
            language_model.add(LSTM(output_dim = arg.lstm_units, return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    else:
        language_model.add(Reshape(input_shape = (arg.language_feature_dim,), dims=(arg.language_feature_dim,)))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode = 'concat', concat_axis = 1))
    if arg.maxout is True:
        for cur_units in arg.mlp_units:
            model.add(MaxoutDense(output_dim = cur_units, nb_feature = 2, init = 'uniform'))
            model.add(Dropout(arg.dropout))
    else:
        for cur_units in arg.mlp_units:
            model.add(Dense(output_dim = cur_units, init = 'uniform'))
            model.add(Activation(arg.mlp_activation))
            model.add(Dropout(arg.dropout))
    model.add(Dense(output_dim = word_vec_dim, init = 'uniform'))
    # model.add(Activation('softmax'))

    print '*** save model ***'
    model_file_name = './models/'
    model_file_name += arg.model_name.replace('_1500_train.csv', '').replace('_1500_test.csv', '')
    open(model_file_name + '.json', 'w').write( model.to_json() )
    # sgd = SGD(lr = arg.lr, decay = 1e-6, momentum = arg.momentum, nesterov = True)
    model.compile(loss = cos_sim, optimizer = 'rmsprop')

    # load data
    print '*** load data ***'
    idMap = loadIdMap()
    answerData = loadAnswerData()
    imageData = loadFeatureData(fileName = './data/image_feature/caffenet_4096_train.csv')
    questionData = loadFeatureData(fileName = arg.question_feature)
    choiceData = loadFeatureData(fileName = arg.choice_feature)

    # training
    print '*** start training ***'
    idList = idMap.keys()
    if arg.cross_valid == 1:
        # fit training
        for i in xrange(arg.epochs):
            print 'epoch #{:03d}'.format(i+1)
            totalloss = 0
            questionIdList, batchNum = prepareIdList(idList, arg.batch_size)
            for j in xrange(batchNum):
                imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
                if arg.language_feature_dim == 1800:
                    loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                      getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                elif arg.language_feature_dim == 300:
                    loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                      getQuestionFeature(questionData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                else:
                    raise Exception("language feature dim error!")
                totalloss += loss[0]
                if (j+1) % 100 == 0:
                    print 'epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}'.format(i+1, j+1, totalloss/(j+1))
            if (i+1) % 5 == 0:
                model.save_weights(model_file_name + '_epoch_{:03d}_loss_{:.3f}.hdf5'.format(i+1, totalloss/batchNum))
    else:
        # cross valid & training
        dataSize = len(idList)
        setSize = dataSize / arg.cross_valid
        for k in xrange(arg.cross_valid):
            # cut train and valid id list
            if k == arg.cross_valid -1:
                validIdList = idList[k * setSize:]
                trainIdList = idList[:k * setSize]
            else:
                validIdList = idList[k * setSize : (k + 1) * setSize]
                trainIdList = idList[:k * setSize] + idList[(k + 1) * setSize:]

            for i in xrange(arg.epochs):
                print 'valid #{02d}, epoch #{:03d}'.format(k+1, i+1)
                # training
                totalloss = 0
                questionIdList, batchNum = prepareIdList(trainIdList, arg.batch_size)
                for j in xrange(batchNum):
                    imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
                    if arg.language_feature_dim == 1800:
                        loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                          getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    elif arg.language_feature_dim == 300:
                        loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                          getQuestionFeature(questionData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    else:
                        raise Exception("language feature dim error!")
                    totalloss += loss[0]
                    if (j+1) % 100 == 0:
                        print 'valid #{:02d}, epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}'.format(k+1, i+1, j+1, totalloss/(j+1))

                # cross valid
                totalerror = 0
                questionIdList, batchNum = prepareIdList(validIdList, 512)
                for j in xrange(batchNum):
                    imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
                    if arg.language_feature_dim == 1800:
                        y_predict = model.predict(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                        getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                  verbose = 0))
                    elif arg.language_feature_dim == 300:
                        y_predict = model.predict(X = [ getImageFeature(imageData, imageIdListForBatch),
                                                        getQuestionFeature(questionData, questionIdList[j]) ],
                                                  verbose = 0 ))
                    else:
                        raise Exception("language feature dim error!")
                print 'valid #{:02d}, epoch #{:03d}, current error = {:.3f}'.format(k+1, i+1, totalerror/len(validIdList))
                if (i+1) % 5 == 0:
                    model.save_weights(model_file_name + '_valid_{:02d}_epoch_{:03d}_loss_{:.3f}.hdf5'.format(k+1, i+1, totalloss/batchNum))