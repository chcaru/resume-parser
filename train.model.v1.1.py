print('Loading dependencies...')

import math, sys, time, json
import numpy as np

from keras.layers import Dense, Dropout, Input, Flatten, Embedding, Bidirectional, TimeDistributed, GRU, Concatenate, Conv1D
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
sequenceLength = options['model']['bufferSize'] * 2 + 1

dataXWordFragments = np.load(options['data']['dataXWordFragments'])
dataXPositionFragments = np.load(options['data']['dataXPositionFragments'])
dataYCategories = np.load(options['data']['dataYCategories'])

# shuffleIndices = np.random.choice(np.arange(len(dataXWordFragments)), len(dataXWordFragments), False)

embeddingMatrixFilePath = options['data']['embeddingMatrix'].replace('{vocabSize}', str(vocabSize - 1))
embeddingMatrix = np.load(embeddingMatrixFilePath)

print('Building model...')

WordEmbedding = Embedding(input_dim=vocabSize,
    output_dim=wordVectorSize,
    weights=[embeddingMatrix],
    trainable=False)

fragmentWordsInput = Input(shape=(sequenceLength,), dtype='int32')
positionInput = Input(shape=(1,), dtype='float32')

vectorizedFragmentWords = WordEmbedding(fragmentWordsInput)

# fragmentWordsEncoder = BGRU(128, rs=False, ur=True)(vectorizedFragmentWords)
fragmentWordsEncoder = Conv1D(128, 2, activation='relu')(vectorizedFragmentWords)
fragmentWordsEncoder = Flatten()(fragmentWordsEncoder)

fragmentWordsPosition = Concatenate()([positionInput, fragmentWordsEncoder])

fragmentWordsPositionEncoder = Dense(64, activation='relu')(fragmentWordsPosition)

tokenCategory = Dense(20, activation='softmax')(fragmentWordsPositionEncoder)

resumeParser = Model([fragmentWordsInput, positionInput], tokenCategory)
resumeParser.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
resumeParser.summary()

numEpochs = 100
batchSize = 1024

dataYCategories = dataYCategories.reshape((dataYCategories.shape[0], 1))

print('Begin training...')
for i in range(numEpochs):

    result = resumeParser.fit([dataXWordFragments, dataXPositionFragments], dataYCategories, epochs=1, batch_size=batchSize, validation_split=0.10)

    resumeParser.save_weights('./weights/v1/' + str(round(result.history['val_acc'][0] * 100, 4)) + '.' + str(i) + '.v2.weights.h5')