import numpy as np
import math as ma
import os
from pyeeg import hfd, pfd
from scipy.io import wavfile as wav
from python_speech_features.sigproc import framesig
from python_speech_features import mfcc, fbank, logfbank
from pandas import DataFrame


def katz(data, n):
    L = np.hypot(np.diff(data), 1).sum()
    d = np.hypot(data - data[0], np.arange(len(data))).max()
    return ma.log10(n) / (ma.log10(d/L) + ma.log10(n))


def get_fractalfeatues(path, csv, label):
    root, dirs, files = next(os.walk(path))
    fractal_features =[]
    features_vector = DataFrame()
    audio_counter = 1
    for file in files:
        rate, data = wav.read(root + '\\' + file)
        frames = framesig(data, 1000, 500)
        for frame in frames:
            frame_features = []
            for i in range(2,12):
                frame_features.append(hfd(frame,i))
            for i in range(2,12):
                frame_features.append(katz(frame,i))
            for i in range(2, 12):
                frame_features.append(pfd(frame, frame+i))
            frame_features.append(label)
            fractal_features.append(frame_features)
        print(str(audio_counter) + '==>' + file)
        audio_counter += 1
    features_vector = DataFrame(fractal_features)
    features_vector.to_csv(csv, index=False, header=False, mode='a')


def get_mfccfeatues(path, csv, label):
    root, dirs, files = next(os.walk(path))
    fractal_features =[]
    features_vector = DataFrame()
    audio_counter = 1
    for file in files:
        rate, data = wav.read(root + '\\' + file)
        frames = framesig(data, 1000, 500)
        for frame in frames:
            frame_features = []
            mfcc_features = mfcc(frame, rate, numcep=30)
            for i in range(0, len(mfcc_features[1])):
                frame_features.append(mfcc_features[1][i])
            frame_features.append(label)
            fractal_features.append(frame_features)
        print(str(audio_counter) + '==>' + file)
        audio_counter += 1
    features_vector = DataFrame(fractal_features)
    features_vector.to_csv(csv, index=False, header=False, mode='a')
    print(np.asarray(features_vector).shape)


def get_fbankfeatues(path, csv, label):
    root, dirs, files = next(os.walk(path))
    fractal_features =[]
    features_vector = DataFrame()
    audio_counter = 1
    for file in files:
        rate, data = wav.read(root + '\\' + file)
        frames = framesig(data, 1000, 500)
        for frame in frames:
            frame_features = []
            FBank= fbank(frame, rate)
            for i in range(0, len(FBank[0][1])):
                frame_features.append(FBank[0][1][i])
            frame_features.append(label)
            fractal_features.append(frame_features)
        print(str(audio_counter) + '==>' + file)
        audio_counter += 1
    features_vector = DataFrame(fractal_features)
    features_vector.to_csv(csv, index=False, header=False, mode='a')
    print(np.asarray(features_vector).shape)


def get_logfbankfeatues(path, csv, label):
    root, dirs, files = next(os.walk(path))
    fractal_features =[]
    features_vector = DataFrame()
    audio_counter = 1
    for file in files:
        rate, data = wav.read(root + '\\' + file)
        frames = framesig(data, 1000, 500)
        for frame in frames:
            frame_features = []
            LogFBank = logfbank(frame, rate)
            for i in range(0, len(LogFBank[0])):
                frame_features.append(LogFBank[0][i])
            frame_features.append(label)
            fractal_features.append(frame_features)
        print(str(audio_counter) + '==>' + file)
        audio_counter += 1
    features_vector = DataFrame(fractal_features)
    features_vector.to_csv(csv, index=False, header=False, mode='a')
    print(np.asarray(features_vector).shape)


if __name__ == '__main__':
    get_logfbankfeatues('D:\\Databases\\Steganalysis\\Normal',
                    'D:\\Databases\\Steganalysis\\Dataset\\LogFBank-Features-steghide-7.csv', 0)

    get_logfbankfeatues('D:\\Databases\\Steganalysis\\steghide\\7',
                        'D:\\Databases\\Steganalysis\\Dataset\\LogFBank-Features-steghide-7.csv', 1)



