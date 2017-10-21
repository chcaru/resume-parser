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

def getFragments(indexedResumeX, resumeIndex):

    numFragments = int(((len(indexedResumeX) - fragmentSize) / sequenceStep) + 1)

    fragmentResumeIndices = []

    for iFragment in range(0, numFragments * sequenceStep, sequenceStep):
        
        fragmentResumeIndices.append(resumeIndex)

    return fragmentResumeIndices

dataIndices = []

for i in range(0, len(dataX)):
    print(i)

    dataX[i] = np.lib.pad(dataX[i][:maxResumeTokenCount], (bufferSize, bufferSize), 'constant', constant_values=(vocabSize + 1, vocabSize + 1))

    categoryFragments = getFragments(dataX[i], i)

    dataIndices.extend(categoryFragments)
    
dataIndices = np.asarray(dataIndices, dtype='int32')

np.save(options['data']['dataIndices'], dataIndices)