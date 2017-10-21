import numpy as np
import re, json

options = json.load(open('config.json'))

bufferSize = options['model']['bufferSize']
sequenceStep = options['model']['sequenceStep']
vocabSize = options['model']['vocabSize']
fragmentSize = bufferSize * 2 + 1
maxResumeTokenCount = options['model']['maxResumeTokenCount']

print('Loading data...')
dataX = np.load(options['data']['prep']['dataXIndexed'])
dataY = np.load(options['data']['prep']['dataYIndexed'])

assert len(dataX) == len(dataY)

def getFragments(indexedResumeX, indexedResumeY, trueResumeSize):

    numFragments = int(((len(indexedResumeX) - fragmentSize) / sequenceStep) + 1)

    wordFragments = []
    positionFragments = []
    categoryFragments = []

    trueResumeSize = len(indexedResumeX) - fragmentSize

    for iFragment in range(0, numFragments * sequenceStep, sequenceStep):
        
        fragmentStart = iFragment
        fragmentEnd = fragmentStart + fragmentSize

        wordFragments.append(np.asarray(indexedResumeX[fragmentStart:fragmentEnd], dtype='int32'))
        positionFragments.append(fragmentStart / trueResumeSize)
        categoryFragments.append(indexedResumeY[fragmentStart + bufferSize])

    return wordFragments, positionFragments, categoryFragments

dataXWordFragments = []
dataXPositionFragments = []
dataYCategories = []

for i in range(0, len(dataX)):
    print(i)

    trueResumeSize = len(dataX[i])

    dataX[i] = np.lib.pad(dataX[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(vocabSize + 1, vocabSize + 1))
    dataY[i] = np.lib.pad(dataY[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(vocabSize + 1, vocabSize + 1))

    wordFragments, positionFragments, categoryFragments = getFragments(dataX[i], dataY[i], trueResumeSize)

    dataXWordFragments.extend(wordFragments)
    dataXPositionFragments.extend(positionFragments)
    dataYCategories.extend(categoryFragments)
    
dataXWordFragments = np.asarray(dataXWordFragments, dtype='int32')
dataXPositionFragments = np.asarray(dataXPositionFragments, dtype='float32')
dataYCategories = np.asarray(dataYCategories, dtype='int32')

np.save(options['data']['dataXWordFragments'], dataXWordFragments)
np.save(options['data']['dataXPositionFragments'], dataXPositionFragments)
np.save(options['data']['dataYCategories'], dataYCategories)