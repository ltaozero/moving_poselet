import numpy as np
from .sequence_3d import pad_sequences_3d, extract_feat
from keras.engine.training import make_batches
def mp_data_generator(X, y, batch_size,data_gen_params):
    nb_train_sample = len(X)
    dataset = data_gen_params['dataset']
    features = data_gen_params['features']
    sample_rate_set = data_gen_params['sample_rate_set']
    window_size = data_gen_params['window_size']
    full_BP = data_gen_params['full_BP']
    padding = data_gen_params['padding']
    compute_vec = data_gen_params['compute_vec']
    maxlen = data_gen_params['maxlen']
    while 1:
        #print("i is: ",idx)
        index_array = np.random.permutation(nb_train_sample)
        batches = make_batches(nb_train_sample, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            y_batch = y[batch_ids]        
            X_batch = [X[i] for i in batch_ids]
            if features=='raw':
                X_BP_batch = [extract_feat(X_batch, full_BP,dataset,sample_rate,window_size = window_size,compute_vec=compute_vec) for sample_rate in sample_rate_set]
            else:
                X_BP_batch = [X_batch]
            if padding:   
                X_BP_batch = [pad_sequences_3d(X_BP_batch[i], value = 1.0, dtype='float32', bias=0) for i in range(len(sample_rate_set))]
        
            yield (X_BP_batch, y_batch)