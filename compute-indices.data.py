import numpy as np
import re, json

options = json.load(open('config.json'))

print('Loading data...')
vocabWords = open(options['data']['vocab'].replace('{vocabSize}', str(options['model']['vocabSize']))).readlines()
indexLookup = { vocabWords[i].strip(): i for i in range(len(vocabWords)) }
notFoundIndex = len(vocabWords)
resumes = np.load(options['data']['prep']['labeled'])

dataX = []
dataY = []

for i in range(0, len(resumes)):
    print(i)

    labeledTokens = resumes[i]['raw']['labeledTokens']

    numTokens = len(labeledTokens)

    resumeDataX = np.zeros((numTokens,), dtype='int32')
    resumeDataY = np.zeros((numTokens,), dtype='int32')

    for iLabeledToken in range(len(labeledTokens)):

        resumeDataX[iLabeledToken] = indexLookup[labeledTokens[iLabeledToken][0]] if labeledTokens[iLabeledToken][0] in indexLookup else notFoundIndex
        resumeDataY[iLabeledToken] = labeledTokens[iLabeledToken][1]
    
    dataX.append(resumeDataX)
    dataY.append(resumeDataY)

np.save(options['data']['prep']['dataXIndexed'], dataX)
np.save(options['data']['prep']['dataYIndexed'], dataY)