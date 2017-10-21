import numpy as np
import json

options = json.load(open('config.json'))

bufferSize = options['model']['bufferSize']
sequenceStep = options['model']['sequenceStep']
vocabSize = options['model']['vocabSize']
fragmentSize = bufferSize * 2 + 1
maxResumeTokenCount = options['model']['maxResumeTokenCount']

print('Loading data...')
dataYPredictedCategoriesArgMax = np.load('./data/results/results.argmax.v1.npy')
dataResumeIndices = np.load(options['data']['dataIndices'])

print('Calculating argmax...')

resumeTokenPredictions = []

iResult = 0
while iResult < len(dataYPredictedCategoriesArgMax):

    tokenPredictions = [dataYPredictedCategoriesArgMax[iResult]]

    iResult += 1

    while iResult < len(dataYPredictedCategoriesArgMax) and dataResumeIndices[iResult-1] == dataResumeIndices[iResult]:
        if iResult % 1000000 == 0: print(iResult)

        tokenPredictions.append(dataYPredictedCategoriesArgMax[iResult])

        iResult += 1

    resumeTokenPredictions.append(np.asarray(tokenPredictions, dtype='int32'))

print('Saving data...')
np.save('./data/results/resume.token.predictions.v1.npy', resumeTokenPredictions)
