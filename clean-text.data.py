import numpy as np
import re, json

options = json.load(open('config.json'))

linkRE = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
spaceLeft = re.compile(r'(?<=\S)([\\\!\#\$\%\^\&\*\(\)\-\_\=\+\]\[\}\\{\|\;\:\'\"\,\.\/\?\@]+)(?=\s|$)')
spaceRight = re.compile(r'(?<!\S)([\\\!\#\$\%\^\&\*\(\)\-\_\=\+\]\[\}\\{\|\;\:\'\"\,\.\/\?\@]+)(?=\S)')
space1OrMoreBoth = re.compile(r'(?<=\S)([\\\!\#\$\%\^\&\*\(\)\-\_\=\+\]\[\}\\{\|\;\:\'\"\,\.\/\?\@]+)(?=\S)')
extraSpaces = re.compile(r'(?<=\S)(\s{2,})(?=\S)')

def splitSpaceJoin(match):
    return ' '.join(list(match.groups()[0]))

def cleanText(resumeText):
    
    resumeText = re.sub(linkRE, 'hyperlink', resumeText)
    resumeText = re.sub(spaceLeft, lambda m: ' %s' % splitSpaceJoin(m), resumeText)
    resumeText = re.sub(spaceRight, lambda m: '%s ' % splitSpaceJoin(m), resumeText)
    resumeText = re.sub(space1OrMoreBoth, lambda m: ' %s ' % splitSpaceJoin(m), resumeText)
    resumeText = re.sub(extraSpaces, ' ', resumeText)

    return resumeText.strip()

resumes = np.load(options['data']['prep']['text'])

for i in range(0, len(resumes)):
    print(i)

    resumes[i]['raw']['text'] = cleanText(resumes[i]['raw']['text'].decode('utf8'))

np.save(options['data']['prep']['cleaned'], resumes)
