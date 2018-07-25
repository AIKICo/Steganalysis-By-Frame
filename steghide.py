import os
from subprocess import Popen, PIPE

root, dirs, files = next(os.walk('D:\\Databases\\Steganalysis\\Outguess\\100'))
secret_filename = '100.txt'
for file in files:
    if file.lower().endswith('.wav'):
        p = Popen('outguess.exe -k "kabinet95" -d D:\\Databases\\Steganalysis\\SecretFile\\' + secret_filename + ' ' +
                  'D:\\Databases\\Steganalysis\\Outguess\\100\\' + file +
                  'D:\\Databases\\Steganalysis\\Outguess\\100\\' + file , shell=True,
                  stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        print(file)
        if err != '':
            print(err)