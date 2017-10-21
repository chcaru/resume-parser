print('Loading dependencies...')

import math, sys, time, json
import numpy as np

from keras.layers import Dense, Dropout, Input, Flatten, Embedding, Bidirectional, TimeDistributed, GRU, Concatenate
from keras.models import Model

np.random.seed(1337)

TD = TimeDistributed
Bi = Bidirectional
_GRU = GRU
GRU = lambda s, rs=True, gb=False, ur=False, t=True: _GRU(s, return_sequences=rs, implementation=2, unroll=ur, go_backwards=gb, trainable=t)
BGRU = lambda s, rs=True, gb=False, ur=False, t=True: Bi(GRU(s, rs, gb, ur, t))

print('Loading data...')

options = json.load(open('config.json'))

vocabSize = options['model']['vocabSize'] + 1
wordVectorSize = options['model']['wordVectorSize']
sequenceLength = 8 * 2 + 1

dataXWordFragments = np.load(options['data']['v2']['dataXWordFragments'])
dataXPredictionFragments = np.load(options['data']['v2']['dataXPredictionFragments'])
dataXPositionFragments = np.load(options['data']['v2']['dataXPositionFragments'])
dataYCategories = np.load(options['data']['v2']['dataYCategories'])

embeddingMatrixFilePath = options['data']['embeddingMatrix'].replace('{vocabSize}', str(vocabSize - 1))
embeddingMatrix = np.load(embeddingMatrixFilePath)

print('Building model...')

WordEmbedding = Embedding(input_dim=vocabSize,
    output_dim=wordVectorSize,
    weights=[embeddingMatrix],
    trainable=False)

PredictionEmbedding = Embedding(input_dim=21, output_dim=21)

fragmentWordsInput = Input(shape=(sequenceLength,), dtype='int32')
fragmentPredictionsInput = Input(shape=(sequenceLength,), dtype='int32')
positionInput = Input(shape=(1,), dtype='float32')

vectorizedFragmentWords = WordEmbedding(fragmentWordsInput)

vectorizedFragmentPredictions = PredictionEmbedding(fragmentPredictionsInput)

fragmentWordPredictions = Concatenate()([vectorizedFragmentWords, vectorizedFragmentPredictions])

fragmentWordsEncoder = BGRU(384, rs=False, ur=True)(fragmentWordPredictions)

fragmentWordsPosition = Concatenate()([positionInput, fragmentWordsEncoder])

fragmentWordsPositionEncoder = Dense(512, activation='relu')(fragmentWordsPosition)

tokenCategory = Dense(20, activation='softmax')(fragmentWordsPositionEncoder)

resumeParser = Model([fragmentWordsInput, fragmentPredictionsInput, positionInput], tokenCategory)
resumeParser.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
resumeParser.summary()

numEpochs = 100
batchSize = 1024

dataYCategories = dataYCategories.reshape((dataYCategories.shape[0], 1))

print('Begin training...')
for i in range(numEpochs):

    result = resumeParser.fit([dataXWordFragments, dataXPredictionFragments, dataXPositionFragments], dataYCategories, epochs=1, batch_size=batchSize, validation_split=0.10)

    resumeParser.save_weights('./weights/v2/' + str(round(result.history['val_acc'][0] * 100, 4)) + '.' + str(i) + '.v1.weights.h5')