import os
from subprocess import Popen, PIPE
import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt

def fn_addnoise(data):
    i = len(data)
    npdata = np.asarray(data).reshape((i))
    u = npdata + np.random.uniform(size=npdata.shape)
    p = npdata + np.random.laplace(loc=0.0, scale=1.0, size=npdata.shape)
    return u,p

if __name__=="__main__":
    root, dirs, files = next(os.walk('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Normal'))

    for file in files:
        if file.lower().endswith('.wav'):
            rate, data = wav.read(root+'\\'+file)
            plt.plot(data)
            wn = np.random.randn(len(data))
            data_wn = data + 0.005*wn
            wav.write('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\NoiseData\\NormalNoise\\'+file, rate, data_wn) 