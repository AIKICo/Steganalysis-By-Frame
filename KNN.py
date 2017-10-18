import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
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


def fractal_modeldata(filename):
    print(filename)
    X, Y = loaddata(filename, 31)
    np.random.seed(13)
    indices = np.random.permutation(2038)
    test_size = int(0.1 * len(indices))
    X_train = X[indices[:-test_size]]
    Y_train = Y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    Y_test = Y[indices[-test_size:]]
    # relu, sigmoid
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    print(accuracy_score(Y_test, Y_pred)*100)
    print(classification_report(Y_test, Y_pred))
    fpr, tpr, thre = roc_curve(Y_test, Y_pred)
    return fpr, tpr, thre


if __name__ == '__main__':
    plt.figure()
    fpr, tpr, thre = fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-hide4pgp-100.csv')
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='100%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    fpr, tpr, thre = fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-hide4pgp-71.csv')
    plt.plot(fpr, tpr, color='lime', lw=1, label='71%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    fpr, tpr, thre = fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-hide4pgp-42.csv')
    plt.plot(fpr, tpr, color='gray', lw=1, label='42%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    fpr, tpr, thre = fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-hide4pgp-21.csv')
    plt.plot(fpr, tpr, color='black', lw=1, label='21%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    fpr, tpr, thre = fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\Fractal\\Fractal-Features-hide4pgp-7.csv')
    plt.plot(fpr, tpr, color='orange', lw=1, label='7%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.show()