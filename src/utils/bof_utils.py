import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy import histogram

def compute_bof_hist(X_BP_train, X_BP_test, K):
    nb_sample = len(X_BP_train)
    data = []
    # first sample data
    for x in X_BP_train:
        x=x[0]
        T = len(x)
        x_sample = shuffle(x,random_state=0)[:T//10]
        data = data + [x]
    data = np.concatenate(data,axis=0)
    model = KMeans(n_clusters=K, random_state=0).fit(data)

    hist_train = []
    hist_test = []
    for x in X_BP_train:
        x=x[0]
        label = model.predict(x)
        hist,_ = histogram(label,bins=range(1,K+1))
        hist_train.append(hist)

    for x in X_BP_test:
        x=x[0]
        label = model.predict(x)
        hist,_ = histogram(label,bins=range(1,K+1))
        hist_test.append(hist)

    return (np.array(hist_train), np.array(hist_test))
