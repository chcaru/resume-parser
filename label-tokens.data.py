import numpy as np
import re, json

options = json.load(open('config.json'))

def getTokens(formattedText):
    return formattedText.strip().split(' ')

def fillEmptyGaps(indexedLabels, fillWithValue):
    
    newIndexedLabels = [indexedLabels[0]]

    for i in range(1, len(indexedLabels)):

        if indexedLabels[i][0] - indexedLabels[i-1][1] > 1:
            newIndexedLabels.append([indexedLabels[i-1][1], indexedLabels[i][0], fillWithValue])

        newIndexedLabels.append(indexedLabels[i])
    
    return newIndexedLabels

def getLabeledTokens(resume):
    
    formattedText = resume['raw']['text']
    indexedLabels = fillEmptyGaps(resume['raw']['textLabels'], 0)
    labeledTokens = []

    for indexedLabel in indexedLabels:

        indexLabel = indexedLabel[2]
        formattedIndex = formattedText[indexedLabel[0]:indexedLabel[1]]
        indexTokens = [[token, indexLabel] for token in getTokens(formattedIndex)]
        labeledTokens.extend(indexTokens)

    return labeledTokens

resumes = np.load(options['data']['prep']['indexed'])

for i in range(0, len(resumes)):
    print(i)

    resumes[i]['raw']['labeledTokens'] = getLabeledTokens(resumes[i])

np.save(options['data']['prep']['labeled'], resumes)
