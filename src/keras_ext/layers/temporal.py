# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer
from keras.regularizers import ActivityRegularizer
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import TimeDistributedDense
#from ..regularizers_ext import WeightRegularizerWithPmask

class TemporalPooling(Layer):
    '''pooling over entire temporal dimension
    input: nb_sample x timesteps x input_dim
    output: nb_sample x input_dim

    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(TemporalPooling, self).__init__(**kwargs)

    def get_output_shape_for(self,input_shape):
        return (input_shape[0], input_shape[2])
   
    def compute_mask(self,x, mask=None):
        return None

    def _pooling_function(self, backend, inputs):
        return NotImplementedError

    # add the masking part
    def call(self, x, mask=None):
        if mask:
            x *= K.cast(K.expand_dims(mask), x.dtype)
        y = self._pooling_function(inputs=x)
        return y

class TemporalSumPooling(TemporalPooling):
    '''sum pooling over entire temporal dimension
    input: nb_sample x timesteps x input_dim
    output: nb_sample x input_dim

    '''
    def __init__(self, **kwargs):
        super(TemporalSumPooling, self).__init__(**kwargs)

    def _pooling_function(self,inputs):
        return K.sum(inputs,axis=-2)

class TemporalAvgPooling(TemporalPooling):
    '''sum pooling over entire temporal dimension
    input: nb_sample x timesteps x input_dim
    output: nb_sample x input_dim

    '''
    def __init__(self, **kwargs):
        super(TemporalAvgPooling, self).__init__(**kwargs)

    def _pooling_function(self,inputs):
        return K.mean(inputs,axis=-2)

class TemporalMaxPooling(TemporalPooling):
    '''max pooling over entire temporal dimension
    input: nb_sample x timesteps x input_dim
    output: nb_sample x input_dim

    '''
    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)

    def _pooling_function(self,inputs):
        return K.max(inputs,axis=-2)



class TimeDistributedDenseWithWmask(TimeDistributedDense):
    '''
       Apply a same Dense layer for each dimension[1] (time_dimension) input.
       Especially useful after a recurrent network with 'return_sequence=True'.
       Tensor input dimensions:   (nb_sample, time_dimension, input_dim)
       Tensor output dimensions:  (nb_sample, time_dimension, output_dim)
       
       Modified: add a mask to W so that certain part of W is not used

    '''
    def __init__(self, output_dim, W_mask=None,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, input_length=None, **kwargs):
         
        super(TimeDistributedDenseWithWmask, self).__init__(output_dim,
                 init, activation, weights,
                 W_regularizer, b_regularizer, activity_regularizer,
                 W_constraint, b_constraint,
                 bias, input_dim, input_length, **kwargs)
        self.W_mask = W_mask

    def call(self, x, mask=None):
        if mask:
            x *= K.cast(K.expand_dims(mask), x.dtype)
        x = x.dimshuffle(1,0,2)
        W = self.W
        if self.W_mask is not None:
            W = self.W*self.W_mask
        y = K.dot(x, W) 
        if self.bias:
            y += self.b*T.neq(x.sum(axis=2,keepdims=True),0)
        y = self.activation(y).dimshuffle(1,0,2)
        return y


class TemporalPyramidMaxPooling(Layer):
    '''max pooling over entire sequence with a temporal pyramid
    input: nb_sample x timesteps x input_dim
    output: nb_sample x (input_dim * # segments)

    '''
    def __init__(self,tp_layer=1, **kwargs):        
        self.tp_layer = tp_layer
        self.supports_masking = True
        super(TemporalPyramidMaxPooling, self).__init__(**kwargs)

    def get_output_shape_for(self,input_shape):
        nseg = np.power(2,self.tp_layer)-1 
        return (input_shape[0], input_shape[2]*nseg)
   
    def compute_mask(self,x, mask):
        return None

    # add the masking part  
    def _step(self,x_tmp,mask_tmp):
        len_x = K.sum(mask_tmp)
        y=[]
        for layer in range(self.tp_layer):
            depth = np.floor(len_x/np.power(2,layer))
            for start_idx in range(np.power(2,layer)):
                o = K.max(x_tmp[start_idx*depth:start_idx*depth+depth,:], axis=-2)
                y = y +[o]
        y = T.concatenate(y, axis=-1)
        return y
   
    # Need to do: change theano.scan to general backend function that works for both theano and tensorflow
    # Lingling Tao, 07/21/2016
    def call(self, x, mask=None):
        if mask:
            x *= K.cast(K.expand_dims(mask), x.dtype)
        if self.tp_layer==1:
            y = K.max(x, axis=-2)
            return y
        if mask:
            y,_ = theano.scan(self._step,
                              sequences=[x,mask])
        else: 
            y = []
            for layer in range(self.tp_layer):
                seg_length = np.floor(X.shape[1]/np.power(2,layer))
                for start_idx in range(np.power(2,layer)):
                    o = K.max(x[:,start_idx*seg_length:start_idx*seg_length+seg_length,:], axis=-2)
                    y = y +[o]
                y = T.concatenate(y, axis=-1)        
        return y

class SubSampling1D(Layer):

    def __init__(self,  subsample_rate=1,**kwargs):
        self.subsample_rate=subsample_rate
        self.supports_masking = True
        super(SubSampling1D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]//self.subsample_rate, input_shape[2])

    def compute_mask(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        output_mask = None
        if mask is not None: 
            output_mask = mask[:,0:input_shape[1]:self.subsample_rate]
        return output_mask        


    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        output = x[:,0: input_shape[1]:self.subsample_rate,:]
        if mask:
            output *= K.cast(K.expand_dims(mask), X.dtype)
        return output
    
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "subsample_rate": self.subsample_rate}
        base_config = super(SubSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedMaxPooling1D(MaxPooling1D):

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(MaskedMaxPooling1D, self).__init__(pool_length, stride, 
                                                 border_mode, **kwargs)

    def compute_mask(self, x, mask=None):
        output_mask = None
        if mask is not None:
            #output_mask = mask[:,0:-self.pool_length+1:self.stride]
            output_mask = mask[:,self.pool_length-1:None:self.stride]
        return output_mask