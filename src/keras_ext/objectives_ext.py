from __future__ import absolute_import
import numpy as np
from keras import backend as K

def multiclass_hinge(y_true, y_pred):
    # convert y_true from 1/0 label to 1/-1 label
    y_true = 2. * y_true - 1.
    # use sum over classes instead of mean
    return K.sum(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
