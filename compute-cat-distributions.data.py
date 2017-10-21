import numpy as np
import re, json

options = json.load(open('config.json'))

bufferSize = 8
sequenceStep = options['model']['sequenceStep']
vocabSize = options['model']['vocabSize']
fragmentSize = bufferSize * 2 + 1
maxResumeTokenCount = options['model']['maxResumeTokenCount']

print('Loading data...')
dataX = np.load(options['data']['prep']['dataXIndexed'])
dataY = np.load(options['data']['prep']['dataYIndexed'])
resumeTokenPredictions = np.load('./data/results/resume.token.predictions.v1.npy')

assert len(dataX) == len(dataY)

def getFragments(indexedResumeX, tokenPredictionsX, indexedResumeY, trueResumeSize):

    numFragments = int(((len(indexedResumeX) - fragmentSize) / sequenceStep) + 1)

    wordFragments = []
    predictionFragments = []
    positionFragments = []
    categoryFragments = []

    trueResumeSize = len(indexedResumeX) - fragmentSize

    for iFragment in range(0, numFragments * sequenceStep, sequenceStep):
        
        fragmentStart = iFragment
        fragmentEnd = fragmentStart + fragmentSize

        wordFragments.append(np.asarray(indexedResumeX[fragmentStart:fragmentEnd], dtype='int32'))
        predictionFragments.append(np.asarray(tokenPredictionsX[fragmentStart:fragmentEnd], dtype='int32'))
        positionFragments.append(fragmentStart / trueResumeSize)
        categoryFragments.append(indexedResumeY[fragmentStart + bufferSize])

    return wordFragments, predictionFragments, positionFragments, categoryFragments

dataXWordFragments = []
dataXPredictionFragments = []
dataXPositionFragments = []
dataYCategories = []

for i in range(0, len(dataX)):
    if i % 100 == 0: print(i)

    assert len(dataX[i]) == len(dataY[i])
    assert len(dataX[i]) == len(resumeTokenPredictions[i]) or (len(dataX[i]) >= maxResumeTokenCount and len(resumeTokenPredictions[i]) == maxResumeTokenCount)

    trueResumeSize = len(dataX[i])

    dataX[i] = np.lib.pad(dataX[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(vocabSize + 1, vocabSize + 1))
    resumeTokenPredictions[i] = np.lib.pad(resumeTokenPredictions[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(20, 20))
    dataY[i] = np.lib.pad(dataY[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(vocabSize + 1, vocabSize + 1))
    
    wordFragments, predictionFragments, positionFragments, categoryFragments = getFragments(dataX[i], resumeTokenPredictions[i], dataY[i], trueResumeSize)

    dataXWordFragments.extend(wordFragments)
    dataXPredictionFragments.extend(predictionFragments)
    dataXPositionFragments.extend(positionFragments)
    dataYCategories.extend(categoryFragments)
    
dataXWordFragments = np.asarray(dataXWordFragments, dtype='int32')
dataXPredictionFragments = np.asarray(dataXPredictionFragments, dtype='int32')
dataXPositionFragments = np.asarray(dataXPositionFragments, dtype='float32')
dataYCategories = np.asarray(dataYCategories, dtype='int32')

np.save(options['data']['v2']['dataXWordFragments'], dataXWordFragments)
np.save(options['data']['v2']['dataXPredictionFragments'], dataXPredictionFragments)
np.save(options['data']['v2']['dataXPositionFragments'], dataXPositionFragments)
np.save(options['data']['v2']['dataYCategories'], dataYCategories)
