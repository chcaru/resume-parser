print('Loading dependencies...')
import gc, math, sys, re, os, json, gc
import numpy as np
from collections import Counter

options = json.load(open('config.json'))

maxVocab = options['model']['vocabSize']

print('Loading data...')
cleanedResumes = np.load(options['data']['prep']['cleaned'])

preTrainedWordEmbeddingsPath = options['glove']
preTrainedWordEmbeddingSize = options['model']['wordVectorSize']

showDroppedItems = False

print('Computing tokens...')
tokens = (' '.join([cleanedResume['raw']['text'].strip() for cleanedResume in cleanedResumes])).split(' ')

print('Indexing tokens...')
uniqueTokens = Counter(tokens)
uniqueTokens[' '] = len(tokens)
tokenLookup = { token: index for token, index in zip(sorted(uniqueTokens, key=uniqueTokens.get, reverse=True), range(len(uniqueTokens))) }
reverseTokenLookup = { index: token for token, index in tokenLookup.items() }

maxVocab = min(maxVocab, len(reverseTokenLookup))

tokens = None
gc.collect()

print('Building word vector table...')
wordVectorVocab = np.load(options['data']['prep']['wordVectorVocab'])
wordVectors = np.load(options['data']['prep']['wordVectors'])
assert len(wordVectorVocab) == len(wordVectors)
preTrainedEmbeddingLookup = { wordVectorVocab[i].decode('utf8'): wordVectors[i] for i in range(len(wordVectors)) }

embeddingMatrix = np.zeros((maxVocab + 1, preTrainedWordEmbeddingSize), dtype='float32')

# fix this to find #maxVocab valid pre trained words embeddings
print('Computing vocab...')
vocabFilePath = options['data']['vocab'].replace('{vocabSize}', str(maxVocab))
vocabFile = open(vocabFilePath, 'w')
iVocab = 0
iEmbeddingMatrix = 0
while iVocab < len(reverseTokenLookup) and iEmbeddingMatrix < maxVocab:

    word = reverseTokenLookup[iVocab]

    wordVector = preTrainedEmbeddingLookup.get(word)

    if wordVector is not None:
        vocabFile.write(reverseTokenLookup[iVocab].strip() + '\n')
        embeddingMatrix[iEmbeddingMatrix + 1] = wordVector
        iEmbeddingMatrix += 1
    else:
        print(reverseTokenLookup[iVocab] + ' was not found in pre trained word embeddings')
    
    iVocab += 1

vocabFile.close()
embeddingMatrixFilePath = options['data']['embeddingMatrix'].replace('{vocabSize}', str(maxVocab))
np.save(embeddingMatrixFilePath, embeddingMatrix)

# This is stuper slow so only use if needed...
# vocabFile = None
# embeddingMatrix = None
# preTrainedEmbeddingLookup = None
# gc.collect()

def wordToIndex(w):
    if w not in tokenLookup:
        return 0
    i = tokenLookup[w]
    return i + 1 if i + 1 < maxVocab else 0

def indexToWord(i):
    return reverseTokenLookup[i-1] if i > 0 else '~'