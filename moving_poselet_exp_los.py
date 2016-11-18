import numpy as np
import sys 
import os
from scipy import io as sio
from keras.utils import np_utils
from src.utils.seq_dataset import load_data
from src.utils.sequence_3d import pad_sequences_3d, extract_feat, create_BP_mask,preprocess_data
from src.models.create_model import create_MP_model
from src.utils.opt_parser import mp_parser, process_params
from src.utils.data_generator import mp_data_generator
#np.random.seed(1337)  # for reproducibility
#sys.setrecursionlimit(50000)


parser = mp_parser()
params = parser.parse_args()
params = vars(params)
data_gen_params = process_params(params)


# change basedir to the folder where data are saved. 
# It should have same format as the provided data in data folder
basedir = '~/work/Data/'
    
# load body part config info, generate mask
dataset = params['dataset']
filename = '{}_result_all2_{}.mat'.format(dataset, params['subset'])

joint_map={'MSR3D':20,'MSRDaily':20,'CompAct':20,'MHAD':35,'HDM05':31,'CAD120':15,'Suturing':4, 'KnotTying':4, 'NeedlePassing':4}
njoints = joint_map[dataset]


print("Loading Data...")
data_generation = False
X_train, y_train, X_test, y_test = load_data(basedir, dataset, data_gen_params['features'], params['subset'])
nb_classes = len(np.unique(y_train))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

if params['features'] == 'raw':
    input_dim = 3*njoints*data_gen_params['window_size']*(data_gen_params['compute_vec']+1)
    input_dims = input_dim*np.ones(len(data_gen_params['sample_rate_set'])) 
else:
    # use input features directly
    input_dim = X_train[0].shape[-1]
    input_dims = [input_dim]

if params['use_fb']:
    W_mask = None
    MP_per_model = params['num_MP']
else:

    W_mask = create_BP_mask(dataset, params['num_MP'], input_dim) 
    MP_per_model = W_mask.shape[-1]

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
    # normalize data
    X_std = np.vstack(X_train).std(0).T
    X_train = [(x-x.mean(0))/X_std for x in X_train]
    X_test = [(x-x.mean(0))/X_std for x in X_test]


    X_BP_train, X_BP_test = preprocess_data(X_train, X_test, data_gen_params)    
    hist = model.fit(X_BP_train, Y_train, batch_size=params['batch_size'], nb_epoch=params['nb_epoch'], validation_data=(X_BP_test, Y_test), verbose=2)
    metric = model.evaluate(X_BP_test,Y_test)
else:
    # TO BE ADDED
    hist = model.fit_generator(mp_data_generator(X_train, Y_train, params['batch_size'],data_gen_params), samples_per_epoch =len(y_train), nb_epoch=params['nb_epoch'], validation_data=mp_data_generator(X_test, Y_test,params['batch_size'],data_gen_params), nb_val_samples= len(y_test),verbose=2)

    print("Test Model...")
    metric = model.evaluate_generator(mp_data_generator(X_test, Y_test,params['batch_size'],data_gen_params),len(y_test))

test_acc = metric[1]
print("The accuracy on test data is: ", test_acc)
weights = model.get_weights()
if W_mask is not None:
    weights[0] = weights[0]*W_mask       

# save final data
sio.savemat(filename,{'params':params, 'data_gen_params':data_gen_params,'weights':weights, 'history':hist.history, 'test_acc':test_acc})
