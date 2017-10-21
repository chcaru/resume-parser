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

notFoundCount = 0
foundCount = 0
def getIndexRange(formattedSuperString, unformattedSubString, categoryIndex, seenTable):
    global notFoundCount
    global foundCount
    formattedSubString = cleanText(unformattedSubString.encode('ascii', 'replace').decode())

    if len(formattedSubString) <= 0:
        return None

    lastOccurenceEndIndex = seenTable[formattedSubString] if formattedSubString in seenTable else 0
    
    # print(formattedSubString, lastOccurenceEndIndex)
    startIndex = formattedSuperString.find(formattedSubString, lastOccurenceEndIndex)
    endIndex = startIndex + len(formattedSubString)

    if startIndex == -1:
        notFoundCount += len(formattedSubString)
        print('Not found count: %d' % notFoundCount)
    else:
        foundCount += len(formattedSubString)
        seenTable[formattedSubString] = endIndex

    return [startIndex, endIndex, categoryIndex] if startIndex > -1 else None

def appendIndexRangeFromOptionalProperty(formattedResumeText, labels, parsedTable, optionalProperty, categoryIndex, seenTable):
    if optionalProperty in parsedTable:
        indexRange = getIndexRange(formattedResumeText, parsedTable[optionalProperty], categoryIndex, seenTable)
        if indexRange != None:
            return labels.append(indexRange)

def appendIndexRangeFromOptionalArrayProperty(formattedResumeText, labels, parsedTable, optionalProperty, categoryIndex, seenTable):
    if optionalProperty in parsedTable:
        indexRanges = [getIndexRange(formattedResumeText, unformattedSubString, categoryIndex, seenTable) for unformattedSubString in parsedTable[optionalProperty]]
        indexRanges = [indexRange for indexRange in indexRanges if indexRange != None]
        return labels.extend(indexRanges)

def getHeaderLabels(parsedResume, formattedResumeText, seenTable):

    parsedHeader = parsedResume['header']

    headerLabels = []

    appendIndexRangeFromOptionalProperty(formattedResumeText, headerLabels, parsedHeader, 'title', 18, seenTable)
    appendIndexRangeFromOptionalProperty(formattedResumeText, headerLabels, parsedHeader, 'subtitle', 16, seenTable)
    appendIndexRangeFromOptionalArrayProperty(formattedResumeText, headerLabels, parsedHeader, 'summary', 17, seenTable)
    appendIndexRangeFromOptionalProperty(formattedResumeText, headerLabels, parsedHeader, 'location', 15, seenTable)
    appendIndexRangeFromOptionalProperty(formattedResumeText, headerLabels, parsedHeader, 'elegibility', 14, seenTable)

    return headerLabels


def getExperienceLabels(parsedResume, formattedResumeText, seenTable):

    parsedExperiences = parsedResume['experience']

    experienceLabels = []

    for parsedExperience in parsedExperiences:
        appendIndexRangeFromOptionalProperty(formattedResumeText, experienceLabels, parsedExperience, 'title', 12, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, experienceLabels, parsedExperience, 'company', 8, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, experienceLabels, parsedExperience, 'location', 10, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, experienceLabels, parsedExperience, 'time', 11, seenTable)
        appendIndexRangeFromOptionalArrayProperty(formattedResumeText, experienceLabels, parsedExperience, 'description', 9, seenTable)

    return experienceLabels

def getEducationLabels(parsedResume, formattedResumeText, seenTable):

    parsedEducations = parsedResume['education']

    educationLabels = []

    for parsedEducation in parsedEducations:
        appendIndexRangeFromOptionalProperty(formattedResumeText, educationLabels, parsedEducation, 'title', 7, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, educationLabels, parsedEducation, 'school', 5, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, educationLabels, parsedEducation, 'location', 4, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, educationLabels, parsedEducation, 'time', 6, seenTable)

    return educationLabels

def getSkillsLabels(parsedResume, formattedResumeText, seenTable):

    parsedSkills = parsedResume['skills']

    skillLabels = []

    for parsedSkill in parsedSkills:
        appendIndexRangeFromOptionalProperty(formattedResumeText, skillLabels, parsedSkill, 'text', 19, seenTable)

    return skillLabels

def getCertificationsLabels(parsedResume, formattedResumeText, seenTable):

    parsedCertifications = parsedResume['certifications']

    certificationLabels = []

    for parsedCertification in parsedCertifications:
        appendIndexRangeFromOptionalProperty(formattedResumeText, certificationLabels, parsedCertification, 'title', 3, seenTable)
        appendIndexRangeFromOptionalProperty(formattedResumeText, certificationLabels, parsedCertification, 'time', 2, seenTable)
        appendIndexRangeFromOptionalArrayProperty(formattedResumeText, certificationLabels, parsedCertification, 'description', 1, seenTable)

    return certificationLabels

def getExtraInfoLabels(parsedResume, formattedResumeText, seenTable):

    if 'extra' in parsedResume:

        parsedExtraInfos = parsedResume['extra']

        extraInfoLabels = []

        for parsedExtraInfo in parsedExtraInfos:
            appendIndexRangeFromOptionalArrayProperty(formattedResumeText, extraInfoLabels, parsedExtraInfo, 'text', 13, seenTable)

        return extraInfoLabels

    else: return []

def getLabels(resume):

    labels = []
    seenTable = {}
    parsedResume = resume['parsed']
    formattedResumeText = resume['raw']['text']
    # print(formattedResumeText)
    labels.extend(getHeaderLabels(parsedResume, formattedResumeText, seenTable))
    labels.extend(getExperienceLabels(parsedResume, formattedResumeText, seenTable))
    labels.extend(getEducationLabels(parsedResume, formattedResumeText, seenTable))
    labels.extend(getSkillsLabels(parsedResume, formattedResumeText, seenTable))
    labels.extend(getCertificationsLabels(parsedResume, formattedResumeText, seenTable))
    labels.extend(getExtraInfoLabels(parsedResume, formattedResumeText, seenTable))

    return labels

resumes = np.load(options['data']['prep']['cleaned'])

for i in range(0, len(resumes)):
    print(i)

    resumes[i]['raw']['textLabels'] = getLabels(resumes[i])

print('Not found: %d/%d' % (notFoundCount, foundCount))

np.save(options['data']['prep']['indexed'], resumes)
