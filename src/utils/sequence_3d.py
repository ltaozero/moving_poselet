from __future__ import absolute_import
import numpy as np
import random
import os
from scipy import io as sio
from .seq_dataset import load_data

def pad_sequences_3d(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.,bias=0):
    """
        Pad list of 2D sequences to a 3D tensor, each sequence is a 2d tensor of form T*d, T being length of seq.
        Pad each sequence to the same length: 
        the length of the longuest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

    """
    lengths = [s.shape[-2] for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    if bias==1:
        x = (np.ones((nb_samples, maxlen,sequences[0].shape[-1]+1)) * value).astype(dtype)
    else:
        x = (np.ones((nb_samples, maxlen,sequences[0].shape[-1])) * value).astype(dtype)
      
    for idx, s in enumerate(sequences):
        if s.ndim==3: 
            s=s[0]
        if truncating == 'pre':
            trunc = s[-maxlen:,]
        elif truncating == 'post':
            trunc = s[:maxlen,]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if bias==1:
            trunc=np.pad(trunc,((0,0),(0,1)),mode='constant', constant_values=(1))
        #print(trunc.shape)
        if padding == 'post':
            x[idx, :len(trunc),:] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):,:] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

def create_BP_mask(dataset, num_MP, feat_dim):
    '''
    Input: list of 2D array of size timesteps x input_dim
    Output: list of 2D array of size timesteps x output_dim, which only takes entry from part_index. For each frame t, [..., x(t-sampling_rate), x(t), x(t+sampling_rate),...], which contains window_size frames, is computed as output for frame t.
    '''
    
    if dataset=='Suturing' or dataset=='KnotTying' or dataset=='NeedlePassing':
        BP_info = [range(0,19),range(19,38), range(38,57),range(57,76), range(0,38),range(38,76),range(0,76)]
        mask_all = []
        for part_index in BP_info:
            mask = np.zeros((76,num_MP),dtype='float32')
            mask[part_index,:] = 1.0
            mask_all.append(mask)
        mask_all = np.concatenate(mask_all, axis=1)
        return mask_all

    file_info = os.path.expanduser("~/work/Data/{}/{}_info.mat".format(dataset,dataset))    
    contents = sio.loadmat(file_info)
    BP_info = contents['config'][0]
    dim_per_frame=60 # 3(xyz) * 20joints
    if dataset=='MHAD':
        dim_per_frame=105 # 3(xyz) * 35 joints
    elif dataset =="HDM05":
        dim_per_frame =93
    elif dataset =="CAD120":
        dim_per_frame=45

    n_frame = feat_dim/dim_per_frame
    mask_all = []
    for part_index in BP_info: 

        part_index = np.int_(part_index[0])
        entry_index = np.concatenate([(part_index-1)*3, (part_index-1)*3+1, (part_index-1)*3+2])
        entry_index = np.sort(entry_index)
        entry_index_all=[]
        for i in range(n_frame):
            entry_index_all = np.append(entry_index_all, [entry_index+dim_per_frame*i])
        entry_index_all=np.int_(np.array(entry_index_all))    
        mask = np.zeros((feat_dim,num_MP),dtype='float32')
        mask[entry_index_all,:]=1.0
        mask_all.append(mask)
    mask_all = np.concatenate(mask_all, axis=1)
    
    return mask_all

 
def extract_feat(sequences, part_index,dataset,sampling_rate, window_size=5,compute_vec=1):
    '''
    Input: list of 2D array of size timesteps x input_dim
    Output: list of 2D array of size timesteps x output_dim, which only takes entry from part_index. For each frame t, [..., x(t-sampling_rate), x(t), x(t+sampling_rate),...], which contains window_size frames, is computed as output for frame t.
    '''
    dim_per_frame=60 # 3(xyz) * 20joints
    if dataset=='MHAD':
        dim_per_frame=105 # 3(xyz) * 35 joints
    elif dataset =='HDM05':
        dim_per_frame=93
    elif dataset=='CAD120':
        dim_per_frame=45

    part_index = np.int_(part_index)
    n_frame = sequences[0].shape[1]/dim_per_frame
    #print(n_frame)o
    #print(part_index)
    entry_index = np.concatenate([(part_index-1)*3, (part_index-1)*3+1, (part_index-1)*3+2])
    entry_index = np.sort(entry_index)
    entry_index_all=[]
    if compute_vec==1:
        n_frame = 2*n_frame
    for i in range(n_frame):
        entry_index_all = np.append(entry_index_all, [entry_index+dim_per_frame*i])
    entry_index_all=np.int_(np.array(entry_index_all))    
    #print(entry_index_all)
    x = []
    for s in sequences:
        # add velocity here
        tmp = s
        #tmp[tmp.sum(axis=1)==0,:] = np.nan
        if compute_vec ==1:
            vec = tmp[1:,:] - tmp[0:-1,:]
            norm1 = np.sqrt(np.sum(vec*vec,axis=1,keepdims=True))
            norm = norm1.ravel()
            vec_weight = 0.6
            if dataset =='MSR3D':
                vec_weight = 0.75
            vec[norm>0,:] = vec_weight*vec[norm>0,:] / norm1[norm>0,:]
            
        if compute_vec ==1:
            tmp = np.concatenate((tmp[1:,:],vec),axis=1)
        tmp1 = [np.array(tmp[j*sampling_rate:-(window_size-1-j)*sampling_rate or None, entry_index_all]) for j in range(window_size)]
        tmp1 = np.concatenate(np.array(tmp1),axis=1)
        # 07/08 add this line to remove 0 frames 
        #tmp1 = tmp1[tmp1.sum(axis=1)!=0,:]
        x.append(np.array([tmp1]))

    return x

def preprocess_data(X_train,X_test,data_gen_params):
    '''
        Generate preprocessed data for given dataset. Defalt one will extract pos+vel of 5 frames from raw skeleton feature
        Output: Feature sequences corresponding to different temporal scales
    '''
    dataset = data_gen_params['dataset']
    features = data_gen_params['features']
    sample_rate_set = data_gen_params['sample_rate_set']
    window_size = data_gen_params['window_size']
    full_BP = data_gen_params['full_BP']
    padding = data_gen_params['padding']
    compute_vec = data_gen_params['compute_vec']
    maxlen = data_gen_params['maxlen']

    # for cross_entropy, use 0-1 label
    #y_train = np_utils.to_categorical(Y_train, nb_classes)
    #y_test = np_utils.to_categorical(Y_test, nb_classes)
    if features == 'raw':
        X_BP_train = [extract_feat(X_train, full_BP,dataset,sample_rate,window_size = window_size,compute_vec=compute_vec) for sample_rate in sample_rate_set]
        X_BP_test = [extract_feat(X_test, full_BP,dataset,sample_rate,window_size = window_size,compute_vec=compute_vec) for sample_rate in sample_rate_set]
    else:
        X_BP_train = [X_train]
        X_BP_test = [X_test]
    if padding:
        X_BP_train = [pad_sequences_3d(X_BP_train[i], value = 1.0,maxlen = maxlen, dtype='float32', bias=0) for i in range(len(sample_rate_set))]
        X_BP_test = [pad_sequences_3d(X_BP_test[i], value = 1.0,maxlen = maxlen, dtype='float32', bias=0) for i in range(len(sample_rate_set))]
    return (X_BP_train, X_BP_test)
