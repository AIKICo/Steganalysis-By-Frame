import os
from subprocess import Popen, PIPE

root, dirs, files = next(os.walk('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\NoiseData\Hide4PGP\\100\\'))
secret_filename = '100.txt'
for file in files:
    if file.lower().endswith('.wav'):
        p = Popen('hide D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\SecretFile\\' + secret_filename + ' ' +
                  'D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\NoiseData\Hide4PGP\\100\\' + file, shell=True,
                  stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        print(file)
        if err != '':
            print(err)