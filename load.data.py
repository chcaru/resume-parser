from bs4 import BeautifulSoup
import numpy as np
import json
import os

options = json.load(open('config.json'))

resumeRoot = options['data']['source']['resumes']
resumeCategories = os.listdir(resumeRoot)

def getParsedResume(category, resumeId):
    print(category + ' ' + resumeId)
    resumePath = resumeRoot + '\\' + category + '\\' + resumeId
    with open(resumePath, encoding='utf8') as resumeFile:
        return json.load(resumeFile)

def getResumeIdsForCategory(category):
    catDir = resumeRoot + '\\' + category
    return os.listdir(catDir)

resumes = []

for resumeCategory in resumeCategories:
    resumeIds = getResumeIdsForCategory(resumeCategory)
    parsedResumes = [getParsedResume(resumeCategory, resumeId) for resumeId in resumeIds]
    resumes.extend(parsedResumes)

np.save(options['data']['prep']['loaded'], resumes)