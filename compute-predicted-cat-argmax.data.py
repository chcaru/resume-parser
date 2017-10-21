import numpy as np
import json

options = json.load(open('config.json'))

bufferSize = options['model']['bufferSize']
sequenceStep = options['model']['sequenceStep']
vocabSize = options['model']['vocabSize']
fragmentSize = bufferSize * 2 + 1
maxResumeTokenCount = options['model']['maxResumeTokenCount']

print('Loading data...')
dataYPredictedCategories = np.load('./data/results/result.v1.npy')

print('Calculating argmax...')

dataYPredictedCategoriesArgMax = np.zeros((len(dataYPredictedCategories),), dtype='int32')

for iResult in range(0, len(dataYPredictedCategories)):
    if iResult % 1000000 == 0: print(iResult)
    dataYPredictedCategoriesArgMax[iResult] = np.argmax(dataYPredictedCategories[iResult])

print('Saving data...')
np.save('./data/results/results.argmax.v1.npy', dataYPredictedCategoriesArgMax)
