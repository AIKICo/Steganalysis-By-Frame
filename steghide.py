import os
from subprocess import Popen, PIPE

C='100'

root, dirs, files = next(os.walk('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\NoiseData\StegHide\\'+C+'\\'))
secret_filename = C+'.txt'
for file in files:
    if file.lower().endswith('.wav'):
        p = Popen('steghide embed -p kabinet95 -ef D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\SecretFile\\' + secret_filename + ' ' +
                  '-cf D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\NoiseData\StegHide\\'+C+'\\' + file, shell=True,
                  stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        print(file)
        if err != '':
            print(err)