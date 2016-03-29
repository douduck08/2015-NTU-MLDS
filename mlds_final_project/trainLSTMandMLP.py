import os
import resource
import csv
import random
import numpy as np
import theano.tensor as T
import argparse
import gzip
import cPickle as pickle
from os.path import basename

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape, Merge, Dense, MaxoutDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

# from spacy.en import English
#img_dim = 4096
word_vec_dim = 300

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ifdim', '--image_feature_dim', type=int, default=0)
    parser.add_argument('-iidim', '--image_input_dim', type=int, default=0)
    parser.add_argument('-lfdim', '--language_feature_dim', type=int, default=0)
    parser.add_argument('-ionly', '--image_only', type=str, default=False)
    parser.add_argument('-lonly', '--language_only', type=str, default=False)
    parser.add_argument('-qf', '--question_feature', type=str, required=True)
    parser.add_argument('-cf', '--choice_feature', type=str, required=True)
    parser.add_argument('-if', '--image_feature', type=str, required=True)
    # mlp setting
    parser.add_argument('-u', '--units', nargs='+', type=int, required=True)
    parser.add_argument('-a', '--activation', type=str, default='softplus')
    parser.add_argument('-odim', '--output_dim', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=1.0)
    # train setting
    parser.add_argument('-memory_limit', type=float, default=6.0)
    parser.add_argument('-cross_valid', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=100)
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
    f = gzip.open(fileName, 'rb')
    return pickle.load(f)

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

def cos_sim_on_one(y_true, y_pred):
    dot = np.sum(y_true * y_pred)
    u = np.sqrt(np.sum(np.square(y_true)))
    v = np.sqrt(np.sum(np.square(y_pred)))
    return 1 - dot / (u * v + 0.0001)

def get_error(y_pred, choiceData, answerData, idList):
    error = 0
    for i in xrange(len(idList)):
        loss = 10
        answer = -1
        for j in xrange(5):
            current_loss = cos_sim_on_one(y_pred[i], choiceData[ idList[i] ][ j*300:(j+1)*300 ])
            if current_loss < loss:
                answer = j
                loss = current_loss
        if answer != answerData[ idList[i] ]:
            error += 1
    return error

if __name__ == '__main__':
    arg = parseArgs()
    # limit_memory(arg.memory_limit * 1e9)  # about 6GB
    max_len = 30
    # wordVectorModel = English()

    # image model
    image_model = Sequential()
    image_model.add(Reshape(input_shape = (arg.image_feature_dim,), dims=(arg.image_feature_dim,)))
    # pass image feature to a matrix before merging with language model
    if arg.image_input_dim != 0:
        image_model.add(Dense(output_dim = arg.image_input_dim, init = 'uniform'))
        image_model.add(Activation('softplus'))

    # language model
    language_model = Sequential()
    language_model.add(Reshape(input_shape = (arg.language_feature_dim,), dims=(arg.language_feature_dim,)))

    # merge model
    if arg.image_only == 'True':
        model = image_model
    elif arg.language_only == 'True':
        model = language_model
    else:
        model = Sequential()
        model.add(Merge([image_model, language_model], mode = 'concat', concat_axis = 1))

    if arg.activation == 'maxout':
        for cur_units in arg.units:
            model.add(MaxoutDense(output_dim = cur_units, nb_feature = 2, init = 'uniform'))
            if arg.dropout < 1:
                model.add(Dropout(arg.dropout))
    else:
        for cur_units in arg.units:
            model.add(Dense(output_dim = cur_units, init = 'uniform'))
            model.add(Activation(arg.activation))
            if arg.dropout < 1:
                model.add(Dropout(arg.dropout))
    model.add(Dense(output_dim = word_vec_dim, init = 'uniform'))

    print '*** save model ***'
    model_file_name = './model/'
    model_file_name += basename(arg.question_feature).replace('_300_train.pkl.gz', '').replace('_300_test.pkl.gz', '')
    model_file_name += '_ionly_{}_lonly_{}_ifdim_{:d}_iidim_{:d}_lfdim_{:d}_dropout_{:.1f}_activation_{}_unit'.format(arg.image_only, 
                                                                                                           arg.language_only, 
                                                                                                           arg.image_feature_dim, 
                                                                                                           arg.image_input_dim, 
                                                                                                           arg.language_feature_dim, 
                                                                                                           arg.dropout,
                                                                                                           arg.activation)
    # units for filename
    for cur_units in arg.units:
        model_file_name += '_{:d}'.format(cur_units)
    open(model_file_name + '.json', 'w').write( model.to_json() )
    # sgd = SGD(lr = arg.lr, decay = 1e-6, momentum = arg.momentum, nesterov = True)
    model.compile(loss = cos_sim, optimizer = 'rmsprop')
    
    weights_save = model.get_weights()

    logfilename = './log/' + model_file_name[8:]
    # bufsize = 0 for unbuf IO
    bufsize = 0
    logfile = open(logfilename, 'w', bufsize)
    logfile.write('valid_set,epoch,train_loss,valid_err\n')

    # load data
    print '*** load data ***'
    idMap = loadIdMap()
    answerData = loadAnswerData()
    imageData = loadFeatureData(fileName = arg.image_feature)
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
                if arg.image_only == 'True':
                    loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                elif arg.language_only == 'True' and arg.language_feature_dim == 1800:
                    loss = model.train_on_batch(X = [ getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                elif arg.language_only == 'True' and arg.language_feature_dim == 300:
                    loss = model.train_on_batch(X = [ getQuestionFeature(questionData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                elif arg.language_feature_dim == 1800:
                    loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                      getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                elif arg.language_feature_dim == 300:
                    loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                      getQuestionFeature(questionData, questionIdList[j]) ],
                                                y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                else:
                    raise Exception("language feature dim error!")
                totalloss += loss[0]
                if (j+1) % 100 == 0:
                    print 'epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}'.format(i+1, j+1, totalloss/(j+1))
                    logfile.write('epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}\n'.format(i+1, j+1, totalloss/(j+1)))
            if (i+1) % 5 == 0:
                model.save_weights(model_file_name + '_epoch_{:05d}_loss_{:.3f}.hdf5'.format(i+1, totalloss/batchNum))
    else:
        # cross valid & training
        dataSize = len(idList)
        setSize = dataSize / arg.cross_valid
        crossvalidList = []
        for k in xrange(arg.cross_valid):
            #reset the weights to initial
            model.set_weights(weights_save)

            # cut train and valid id list
            if k == arg.cross_valid -1:
                validIdList = idList[k * setSize:]
                trainIdList = idList[:k * setSize]
            else:
                validIdList = idList[k * setSize : (k + 1) * setSize]
                trainIdList = idList[:k * setSize] + idList[(k + 1) * setSize:]
            
            # for save avg totalerr in each cross validation
            totalerror = 0
            for i in xrange(arg.epochs):
                #print 'valid #{:02d}, epoch #{:03d}'.format(k+1, i+1)
                #logfile.write('valid #{:02d}, epoch #{:03d}\n'.format(k+1, i+1))
                # training
                totalloss = 0
                questionIdList, batchNum = prepareIdList(trainIdList, arg.batch_size)
                for j in xrange(batchNum):
                    imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
                    if arg.image_only == 'True':
                        loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    elif arg.language_only == 'True' and arg.language_feature_dim == 1800:
                        loss = model.train_on_batch(X = [ getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    elif arg.language_only == 'True' and arg.language_feature_dim == 300:
                        loss = model.train_on_batch(X = [ getQuestionFeature(questionData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    elif arg.language_feature_dim == 1800:
                        loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                          getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    elif arg.language_feature_dim == 300:
                        loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                          getQuestionFeature(questionData, questionIdList[j]) ],
                                                    y = getAnswerFeature(choiceData, answerData, questionIdList[j]) )
                    else:
                        raise Exception("language feature dim error!")
                    totalloss += loss[0]
                    #if (j+1) % 100 == 0:
                    #    print 'train #{:02d}, epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}'.format(k+1, i+1, j+1, totalloss/(j+1))
                    #    logfile.write('train #{:02d}, epoch #{:03d}, batch #{:03d}, current avg loss = {:.3f}\n'.format(k+1, i+1, j+1, totalloss/(j+1)))

                # The batchNum will be changed
                totalloss = totalloss / batchNum             

                # cross valid
                totalerror = 0
                questionIdList, batchNum = prepareIdList(validIdList, 512)
                for j in xrange(batchNum):
                    imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
                    y_predict = None
                    if arg.image_only == 'True':
                        y_predict = model.predict(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim) ],
                                                  verbose = 0)
                    elif arg.language_only == 'True' and arg.language_feature_dim == 1800:
                        y_predict = model.predict(X = [ getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                  verbose = 0)
                    elif arg.language_only == 'True' and arg.language_feature_dim == 300:
                        y_predict = model.predict(X = [ getQuestionFeature(questionData, questionIdList[j]) ],
                                                  verbose = 0 )
                    elif arg.language_feature_dim == 1800:
                        y_predict = model.predict(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                        getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                                  verbose = 0)
                    elif arg.language_feature_dim == 300:
                        y_predict = model.predict(X = [ getImageFeature(imageData, imageIdListForBatch, arg.image_feature_dim),
                                                        getQuestionFeature(questionData, questionIdList[j]) ],
                                                  verbose = 0 )
                    else:
                        raise Exception("language feature dim error!")
                    totalerror += get_error(y_predict, choiceData, answerData, questionIdList[j])
                totalerror = 1.0 * totalerror / len(validIdList)
                #print 'valid #{:02d}, epoch #{:03d}, current error = {:.3f}'.format(k+1, i+1, totalerror)
                #logfile.write('valid #{:02d}, epoch #{:03d}, current error = {:.3f}\n'.format(k+1, i+1, totalerror))
                print 'valid #{:02d}, epoch #{:03d}, train loss = {:.3f}, valid error = {:.3f}'.format(k+1, i+1, totalloss, totalerror)

                # log format: "#valid set", "#epoch", "train loss", "valid error"
                logfile.write('{:02d},{:03d},{:.3f},{:.3f}\n'.format(k+1, i+1, totalloss, totalerror))
                
                # save model
                if (i+1) % 20 == 0:
                    model.save_weights(model_file_name + '_valid_{:02d}_epoch_{:03d}_loss_{:.3f}_error_{:.3f}.hdf5'.format(k+1, i+1, totalloss, totalerror))

            # save current cross validation error
            crossvalidList.append(totalerror)
        crossvalidAVG = sum(crossvalidList) / len(crossvalidList)
        print 'AVG. error = {:.3f}\n'.format(crossvalidAVG)
        #logfile.write('AVG. error = {:.3f}\n'.format(crossvalidAVG))
        os.rename(logfilename, logfilename + '_AVGerr_' + '{:.3f}.log'.format(crossvalidAVG))
