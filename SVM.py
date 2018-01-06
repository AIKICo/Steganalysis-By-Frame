import csv
import numpy as np
from sklearn import svm
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)


def fractal_modeldata(filename, inc, plotlabel):
    scores = []
    print(filename)
    X, Y = loaddata(filename, inc)
    np.random.seed(13)
    indices = np.random.permutation(2038)
    test_size = int(0.5 * len(indices))
    X_train = X[indices[:-test_size]]
    Y_train = Y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    Y_test = Y[indices[-test_size:]]
    classifier = svm.SVC(probability=True)

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    y_pred_prob = classifier.predict_proba(X_test)[::, 1]
    fpr, tpr, thre = roc_curve(Y_test, y_pred_prob)
    plt.plot(fpr, tpr,  markerfacecolor='none', label=plotlabel)
    plt.legend()


if __name__ == '__main__':
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-steghide-100.csv',33, 'Fractal')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\DeltaMFCC\\deltaMFCC-Features-steghide-100.csv', 26, 'MFCC')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'MFB')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\FBank\\FBank-Features-steghide-100.csv',33, 'LPC')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\MFCC\\MFCC-Features-steghide-100.csv',33, 'Wavelet')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LPC\\lpc-Features-steghide-100.csv',33, '△MFCC')

    plt.show()





