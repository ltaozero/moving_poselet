import sys
import os
import numpy as np
from scipy import io as sio

def load_data(basedir,dataset,features, sub=1, fold='subject'):

    
    if features =='raw':
        raw_feature_file = os.path.expanduser("{}/{}/{}_Feature_norm.mat".format(basedir,dataset, dataset))
        if dataset =="CAD120":
            raw_feature_file = os.path.expanduser("{}/{}/{}_kin_Feature_norm.mat".format(basedir,dataset, dataset))           
        print(raw_feature_file)
    else:
        raw_feature_file = os.path.expanduser("{}/{}/{}_{}.mat".format(basedir,dataset, dataset,features))
    contents = sio.loadmat(raw_feature_file)
    Y_all = contents["label"].ravel()
    if setup == 'subject':
        partition = contents["subject"].ravel()
    elif setup == 'supertrial':
        partition = contents["trial"]

    if dataset == "MSR3D" or dataset == "MSRDaily":
        trainpartition = [1,3,5,7,9]
        testpartition =  [2,4,6,8,10]
        #trainpartition = [1,2,3,4,5]
        #testpartition=[6,7,8,9,10]
    elif dataset == "MHAD":
        trainpartition = [2,3,5,9]
        testpartition = range(10,15)
    else:
        # all else are leaving one partition out cross validation
        trainpartition = [ i in np.unique(partition) if i !=sub]
        testpartitoin = [sub]


    X_all1 = contents["features"].ravel()
    X_all = []
    for x in X_all1:
        x = x.transpose()
        x[np.isnan(x)]=0
        x = x.astype('float32')
        # add this line 07/08 remove 0 frames
        #x = x[x.sum(axis=1)!=0,:]
        #x = x[np.isnan(x).sum(axis=1)==0,:]
        #vec = x[1:,:] - x[0:-1,:]
        #norm1 = np.sqrt(np.sum(x*x,axis=1,keepdims=True))
        #x = x/norm1
        #norm = norm1.ravel()
        #vec[norm>1e-20,:] = 0.75*vec[norm>1e-20,:] / norm1[norm>1e-20,:]
        #X_all.append(np.concatenate((x[1:,:],vec),axis=1))
        X_all.append(x)

    trainidx = [i for i in range(len(partition)) if partition[i] in trainpartition]
    testidx = [i for i in range(len(partition)) if partition[i] in testpartition]
    
    #remove bad examples
    if dataset=='MSR3D':
        trainidx = np.setdiff1d(np.array(trainidx),np.array([ 339, 340, 350, 351, 352, 357, 360, 534, 535, 536, 546, 551, 552]))
    elif dataset =='MSRDaily':
        trainidx = np.setdiff1d(np.array(trainidx),np.array([ 92, 251, 252, 256, 258]))
    print("Train Partitions:", trainpartition)
    print("Test Partitions:", testpartition)
    print('Training id:',trainidx)
    print('Test id: ', testidx)    
    
    # data for train and test
    X_train = [X_all[i] for i in trainidx]
    X_test = [X_all[i] for i in testidx]
    y_train = [Y_all[i] for i in trainidx]
    y_test = [Y_all[i] for i in testidx]
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    # map action labels to 1:length(unique(labels))
    unique_labels = np.unique(y_train)
    label_map = {unique_labels[i]:i for i in range(len(unique_labels))}
    print label_map
    y_train = [label_map[y_train[i]] for i in range(len(y_train))]
    y_test = [label_map[y_test[i]] for i in range(len(y_test))]
    return (X_train, y_train, X_test, y_test)
