import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score



def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'), delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:len(row)-1])
        y.append(row[-1])
    print(np.asarray(x).shape)
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)


def fractal_modeldata(filename):
    print(filename)
    X, Y = loaddata(filename, 5)
    np.random.seed(13)
    indices = np.random.permutation(1120)
    test_size = int(0.9 * len(indices))
    X_train = X[indices[:-test_size]]
    Y_train = Y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    Y_test = Y[indices[-test_size:]]
    # relu, sigmoid
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)    
    print(classification_report(Y_test, Y_pred))
    fpr, tpr, thre = roc_curve(Y_test, Y_pred)
    print(accuracy_score(Y_test, Y_pred)*100)
    print(roc_auc_score(Y_test, Y_pred))
    return fpr, tpr, thre


if __name__ == '__main__':
    root = 'D:\\\MySourceCodes\\Projects-Python\\Steganalysis-By-Frame\\SteganalysisDatasets\\Dataset\Fractal\\'
    fractal_modeldata(root + 'noisywavlet-Features-steghide-100.csv')
    fractal_modeldata(root + 'noisywavlet-Features-steghide-71.csv')
    fractal_modeldata(root + 'noisywavlet-Features-steghide-42.csv')
    fractal_modeldata(root + 'noisywavlet-Features-steghide-21.csv')
    fractal_modeldata(root + 'noisywavlet-Features-steghide-7.csv')
