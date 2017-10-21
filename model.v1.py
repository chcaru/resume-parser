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
sequenceLength = options['model']['bufferSize'] * 2 + 1

dataXWordFragments = np.load(options['data']['dataXWordFragments'])
dataXPositionFragments = np.load(options['data']['dataXPositionFragments'])

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

fragmentWordsEncoder = BGRU(384, rs=False, ur=True)(vectorizedFragmentWords)

fragmentWordsPosition = Concatenate()([positionInput, fragmentWordsEncoder])

fragmentWordsPositionEncoder = Dense(512, activation='relu')(fragmentWordsPosition)

tokenCategory = Dense(20, activation='softmax')(fragmentWordsPositionEncoder)

resumeParser = Model([fragmentWordsInput, positionInput], tokenCategory)
resumeParser.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
resumeParser.load_weights('./weights/v2/68.0044.0.v1.weights.h5')
resumeParser.summary()

batchSize = 2048

result = resumeParser.predict([dataXWordFragments, dataXPositionFragments], batch_size=batchSize, verbose=1)

np.save('./data/results/result.v1.npy', result)