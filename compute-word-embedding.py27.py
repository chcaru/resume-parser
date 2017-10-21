import numpy as np
import json

options = json.load(open('config.json'))

print 'Reading glove'
wordEmbeddings = open('./data/glove/glove.840B.300d.txt', 'r').readlines()

print 'Parsing glove'
wordEmbeddings = [wordEmbedding.split(' ') for wordEmbedding in wordEmbeddings]

print 'Writing words'
words = [wordEmbedding[0] for wordEmbedding in wordEmbeddings]
np.save(options['data']['prep']['wordVectorVocab'], words)

print 'Building numpy array'
wordVectors = np.asarray([wordEmbedding[1:] for wordEmbedding in wordEmbeddings], dtype='float32')

print 'Saving numpy array'
np.save(options['data']['prep']['wordVectors'], wordVectors)