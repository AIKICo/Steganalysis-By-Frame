import os
from subprocess import Popen, PIPE

root, dirs, files = next(os.walk('D:\\Databases\\Steganalysis\\H4PGP\\7'))
secret_filename = '7.txt'
for file in files:
    if file.lower().endswith('.wav'):
        p = Popen('hide D:\\Databases\\Steganalysis\\SecretFile\\' + secret_filename + ' ' +
                  'D:\\Databases\\Steganalysis\\H4PGP\\7\\' + file, shell=True,
                  stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        print(file)
        if err != '':
            print(err)