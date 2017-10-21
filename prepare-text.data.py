from bs4 import BeautifulSoup
import numpy as np
import codecs, json

def getResumeText(parsedResume):
    html = parsedResume['raw']['html']
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(' ').replace('\xa0', ' ').encode('ascii', 'replace')

resumes = np.load(options['data']['prep']['loaded'])

for i in range(0, len(resumes)):
    print(i)

    resumes[i]['raw']['text'] = getResumeText(resumes[i])

np.save(options['data']['prep']['text'], resumes)