def katz(data, n):
    L = np.hypot(np.diff(data), 1).sum()
    d = np.hypot(data - data[0], np.arange(len(data))).max()
    return ma.log10(n) / (ma.log10(d/L) + ma.log10(n))

def hfd(X, Kmax):
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(numpy.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(numpy.log(numpy.mean(Lk)))
        x.append([numpy.log(float(1) / k), 1])

    (p, r1, r2, s) = numpy.linalg.lstsq(x, L)
    return p[0]

def pfd(X, D=None):
    if D is None:
        D = numpy.diff(X)
        D = D.tolist()
    N_delta = 0
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return numpy.log10(n) / (
        numpy.log10(n) + numpy.log10(n / n + 0.4 * N_delta)
    )

def getfeatures:
    for frame in frames:
            frame_features = []
            for i in range(2,12):
                frame_features.append(hfd(frame,i))
            for i in range(2,12):
                frame_features.append(katz(frame,i))
            for i in range(2, 12):
                frame_features.append(pfd(frame, frame+i))

def Classifier:
    train = getfeatues(Waves_train)
    evaluatet = getfeatures(Waves_Evaluate)
    test = getfeatures(Waves_test)
    CNN = CreateCNNModel()
    CNN.Learn(train)
    CNN.Evaluate(evaluatet)
    CNN.Test(test)