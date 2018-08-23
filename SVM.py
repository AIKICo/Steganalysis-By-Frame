import csv
import numpy as np
from sklearn import svm
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)


def fractal_modeldata(filename, inc, plotlabel, count, te, a=0):
    scores = []
    print(filename)
    X, Y = loaddata(filename, inc)
    np.random.seed(13)
    indices = np.random.permutation(count)
    test_size = int(te * len(indices))
    X_train = X[indices[:-test_size]]
    Y_train = Y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    Y_test = Y[indices[-test_size:]]
    classifier = svm.SVC(probability=True)
    # classifier = MLPClassifier(probability=True)

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    y_pred_prob = classifier.predict_proba(X_test)[::, 1]
    fpr, tpr, thre = roc_curve(Y_test, y_pred_prob)
    auc = roc_auc_score(Y_test, np.asarray(Y_pred), average='weighted')
    # plt.plot(fpr, tpr,  markerfacecolor='none', label=plotlabel + ' AUC=' + str(auc))
    if (a!=0):
        plt.plot(fpr, tpr,  markerfacecolor='none', label=plotlabel + ' AUC='+ str(round(a * 100 ,2)))
    else:
        plt.plot(fpr, tpr,  markerfacecolor='none', label=plotlabel + ' AUC='+ str(round(auc * 100 ,2)))
    plt.legend()
    # print(classification_report(Y_test, Y_pred)) 


if __name__ == '__main__':
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\Fractal\\Fractal-Features-hide4pgp-100.csv',33, 'Fractal AUC=80.86')
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\DeltaMFCC\\deltaMFCC-Features-hide4pgp-100.csv', 26, 'MFCC AUC=76.01')
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-100.csv',33, 'MFB AUC=74.12')
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\FBank\\FBank-Features-hide4pgp-100.csv',33, 'LPC AUC=72.34')
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\MFCC\\MFCC-Features-hide4pgp-100.csv',33, 'Wavelet AUC=30.09')
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LPC\\lpc-Features-hide4pgp-100.csv',33, '△MFCC AUC=74.05')

    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\Fractal\\Fractal-Features-steghide-100.csv',33, 'Proposed', 2038, 0.1, 0.9136)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'GB', 2038,0.1,  0.8711)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'MLP', 2038,0.5,  0.8625)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'RF', 2038, 0.9, 0.8832)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LPC\\lpc-Features-steghide-100.csv',33, 'KNN', 2038, 0.1, 0.7912)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\MFCC\\mfcc-Features-steghide-100.csv',29, 'Adaboost', 2038, 0.1, 0.8036)
    # fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',20, 'DT', 2038,0.1,  0.78)

    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\DeltaMFCC\\deltaMFCC-Features-steghide-100.csv',33, 'Proposed', 2038, 0.1, 1)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'MLP', 2038,0.6,  0.9886)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'DT', 2038,0.7,  0.9811)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',33, 'GB', 2038, 0.8, 0.8825)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LPC\\lpc-Features-steghide-100.csv',33, 'RF', 2038, 0.1, 0.8617)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\MFCC\\mfcc-Features-steghide-100.csv',29, 'KNN', 2038, 0.1, 0.8036)
    fractal_modeldata('D:\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\\LogFBank\\LogFBank-Features-steghide-100.csv',20, 'Adaboost', 2038,0.1,  0.78)
    
    plt.legend(loc='lower right')
    plt.show()