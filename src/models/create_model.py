import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adagrad,Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2, l1l2, WeightRegularizer
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Masking, Flatten, Merge, Reshape,TimeDistributedDense, Permute
from ..keras_ext.layers.temporal import TimeDistributedDenseWithWmask, TemporalPyramidMaxPooling
from ..keras_ext.regularizers_ext import WeightRegularizerWithPmask
from ..keras_ext.optimizers_ext import SGD_step_decay
from ..keras_ext.objectives_ext import multiclass_hinge

def get_optimizer(params):
    
    learning_rate = params['learning_rate']
    sgd1 = SGD_step_decay(lr = learning_rate, decay_block = params['decay_block']) 
    sgd2 = SGD(lr = learning_rate, decay = 1e-5)
    adagrad = Adagrad(lr=learning_rate)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_methods=[sgd1,sgd2,adagrad,adam] 
    optimizer=opt_methods[params['opt_method']]

    return optimizer

def create_MP_model(params,input_dims,W_mask=None):
    
    num_model = len(input_dims) 
    reg_weight = params['reg_weight']
    # Create Model
    print('Start Build Model...')
    model_BP = [Sequential() for i in range(num_model)]
           
    for i in range(num_model):
        model_BP[i].add(Masking(mask_value=1.0,input_shape=[None, input_dims[i]]))
        model_BP[i].add(TimeDistributedDenseWithWmask(params['MP_per_model'], input_dim=input_dims[i], W_mask=W_mask, W_regularizer=WeightRegularizerWithPmask(l2=params['reg_weight'],p_mask=W_mask),init='lecun_uniform',activation='relu'))
        #model_BP[i].add(MaskedMaxPooling1D(pool_length = 100, stride =10))
        model_BP[i].add(TemporalPyramidMaxPooling(tp_layer=params['tp_layer']))

    if (num_model > 1):
        model = Sequential()
        model.add(Merge(model_BP,mode='concat',concat_axis = -1))
    else:
        model = model_BP[0]    
    
    l1_weight = params['l1_alpha']*reg_weight
    l2_weight = (1-params['l1_alpha'])*reg_weight
    model.add(Dense(params['nb_classes'], W_regularizer=l1l2(l1=l1_weight, l2=l2_weight)))

    model.compile(loss=multiclass_hinge, optimizer=get_optimizer(params),metrics=["accuracy"])
    return model
