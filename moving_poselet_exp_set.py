import numpy as np
import sys 
import os
from scipy import io as sio
from keras.utils import np_utils
import src.utils.seq_dataset
from src.utils.sequence_3d import pad_sequences_3d, extract_feat, create_BP_mask,preprocess_data
from src.models.create_model import create_MP_model
from src.utils.opt_parser import mp_parser, process_params
#np.random.seed(1337)  # for reproducibility
#sys.setrecursionlimit(50000)


parser = mp_parser()
params = parser.parse_args()
params = vars(params)
data_gen_params = process_params(params)

dataset = params['dataset']
if dataset =='MSR3D' or dataset == 'MSRDaily':
    subset = [1]
elif dataset =='CompAct':
    subset = range(1,15)

# change basedir to the folder where data are saved. 
# It should have same format as the provided data in data folder
basedir = '/home-3/ltao4@jhu.edu/work/Data/'
filename = '/scratch/users/ltao4@jhu.edu/mp_journal/{}/{}/nword{}_lr{}_obj{}_opt{}_decay{}_l1{}_reg{}_layer{}_rs{}_multi{}.mat'.format(dataset,params['exp_name'],params['num_MP'],params['learning_rate'],'hinge', params['opt_method'],params['decay'],params['l1_alpha'],params['reg_weight'],params['tp_layer'],params['rs'],params['multi_ts'])
    
# load body part config info, generate mask
joint_map={'MSR3D':20,'MSRDaily':20,'CompAct':20,'MHAD':35,'HDM05':31,'CAD120':15}
njoints = joint_map[dataset]
input_dim = 3*njoints*data_gen_params['window_size']*(data_gen_params['compute_vec']+1)
input_dims = [input_dim] 
full_BP = np.arange(njoints)+1
W_mask = create_BP_mask(dataset, params['num_MP'], input_dim) 
MP_per_model = W_mask.shape[-1]
if params['use_fb']:
    W_mask = None
    MP_per_model = num_MP

weights_all=[]
hist_all = []
predictions = []
for sub in subset:
    print("Loading Data...")
    data_generation = False
    X_BP_train, y_train, X_BP_test, y_test = preprocess_data(basedir, dataset, sub, data_gen_params)
    nb_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # add parameters
    decay_block = params['decay']*len(y_train)//params['batch_size']
    params.update({'nb_classes':nb_classes, 'decay_block':decay_block,'maxlen': data_gen_params['maxlen'], 'MP_per_model':MP_per_model})
    print(data_gen_params)
    print(params)

    print("Create Model...")
    model = create_MP_model(params, input_dims, W_mask=W_mask)
    #model = create_motif_model(params, input_dims, W_mask=W_mask)

    print("Train Model...")
    if data_generation is False:        
        hist = model.fit(X_BP_train, Y_train, batch_size=params['batch_size'], nb_epoch=params['nb_epoch'], validation_data=(X_BP_test, Y_test), verbose=2)
    else:
        # TO BE ADDED
        print("not implemented yet")

    print("Test Model...")
    p_test = model.predict_classes(X_BP_test, batch_size=params['batch_size'],verbose=2)
    test_acc = np.mean(p_test == y_test)*100
    print("The accuracy on test data is: ", test_acc)
    weights = model.get_weights()
    weights_all += [weights]
    hist_all += [hist.history]
    predictions += [p_test]

# save final data
sio.savemat(filename,{'weights_all':weights_all,'hist_all': hist_all,'predictions':predictions})
