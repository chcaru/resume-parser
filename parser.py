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
vocabWords = open(options['data']['vocab'].replace('{vocabSize}', str(options['model']['vocabSize']))).readlines()
indexLookup = { i: vocabWords[i].strip() for i in range(len(vocabWords)) }
indexLookup[len(vocabWords)] = ''

categories = {
    0: 'none',
    1: 'certification.description',
    2: 'certification.time',
    3: 'certification.title',
    4: 'education.location',
    5: 'education.school',
    6: 'education.time',
    7: 'education.title',
    8: 'experience.company',
    9: 'experience.description',
    10: 'experience.location',
    11: 'experience.time',
    12: 'experience.title',
    13: 'extra.text',
    14: 'header.elegibility',
    15: 'header.location',
    16: 'header.subtitle',
    17: 'header.summary',
    18: 'header.title',
    19: 'skill.text',
}

def lex(i):
    tokens = np.asarray([indexLookup[x] for x in dataX[i]]).reshape((len(dataX[i]), 1))
    tokenCategories = np.asarray([categories[x] for x in resumeTokenPredictions[i]]).reshape((len(dataX[i]), 1))
    lexemes = np.concatenate((tokens, tokenCategories), axis=-1)
    return lexemes

def parse(i):
    lexemes = lex(i)
    currentIndex = 0

    def valOrDefault(obj, prop, default=None):
        if default == None: default = []
        if prop in obj:
            return obj[prop]
        else:
            obj[prop] = default
            return obj[prop]

    def check(expected):
        return currentIndex < len(lexemes) and (lexemes[currentIndex][1] == expected or lexemes[currentIndex][1].split('.')[0] == expected)

    def match(expected):
        nonlocal currentIndex
        assert check(expected)
        current = lexemes[currentIndex]
        currentIndex += 1
        return current[0]

    def genList(whileType, takeFunc=None, extendFunc=None):
        if takeFunc == None: takeFunc = lambda: match(whileType)
        if extendFunc == None: extendFunc = lambda: genList(whileType, takeFunc)
        if check(whileType):
            take = [takeFunc()]
            extend = extendFunc()
            return take + extend if len(extend) > 0 else take
        else:
            return []

    def joinList(whileType, delimeter=' '):
        return delimeter.join(genList(whileType))

    def obj(objType, members, o=None):
        if o == None: o = {}
        for member in members:
            nones()
            if check(objType + '.' + member):
                take = joinList(objType + '.' + member)
                if len(take) > 0:
                    valOrDefault(o, member).append(take)
            nones()
        if check(objType):
            return obj(objType, members, o)
        else:
            return o

    def checkAndValOrDefaultWith(o, checkType, do):
        if check(checkType):
            val = valOrDefault(o, checkType)
            it = do()
            if len(it) > 0:
                val.append(it)

    def nones():
        return joinList('none')

    def skill():
        return joinList('skill.text')

    def skills():
        return genList('skill.text', lambda: skill(), lambda: skills())

    def extra():
        return joinList('extra.text')

    def header():
        return obj('header', ['elegibility', 'location', 'subtitle', 'summary', 'title'])

    def experience():
        return obj('experience', ['company', 'description', 'location', 'time', 'title'])

    def education():
        return obj('education', ['location', 'school', 'time', 'title'])

    def certification():
        return obj('certification', ['description', 'time', 'title'])

    def resume(o=None):
        if o == None: o = {}
        lastIndex = currentIndex
        checkAndValOrDefaultWith(o, 'header', lambda: header())
        checkAndValOrDefaultWith(o, 'experience', lambda: experience())
        checkAndValOrDefaultWith(o, 'education', lambda: education())
        checkAndValOrDefaultWith(o, 'certification', lambda: certification())
        checkAndValOrDefaultWith(o, 'skills', lambda: skills())
        checkAndValOrDefaultWith(o, 'extra', lambda: extra())
        nones()
        if currentIndex < len(lexemes) and lastIndex != currentIndex:
            return resume(o)
        else:
            return o

    return resume()

def parseAndWrite(i):
    parsed = parse(i)
    with open(str(i) + '.json', 'w') as f:
        json.dump(parsed, f)
    return parsed