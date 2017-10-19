import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve


def loaddata(filename,instanceCol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instanceCol])
        y.append(row[-1])
    return np.array(x[1:]).astype((np.float32)), np.array(y[1:]).astype(np.int)


def fractal_modeldata(filename):
    scores = []
    print(filename)
    X, Y = loaddata(filename, 27)
    np.random.seed(13)
    indices = np.random.permutation(2038)
    test_size = int(0.1 * len(indices))
    X_train = X[indices[:-test_size]]
    Y_train = Y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    Y_test = Y[indices[-test_size:]]
    model = Sequential()
    model.add(Dense(512, input_dim=27, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=0)
    score = model.evaluate(X_test, Y_test, batch_size=16)
    classes = model.predict_classes(X_test, batch_size=1)
    print(score)
    print(accuracy_score(Y_test, np.asarray(classes))*100)
    print(classification_report(Y_test, np.asarray(classes)))


if __name__ == '__main__':
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-100.csv')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-71.csv')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-42.csv')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-21.csv')
    fractal_modeldata('D:\\Databases\\Steganalysis\\Dataset\\LogFBank\\LogFBank-Features-hide4pgp-7.csv')