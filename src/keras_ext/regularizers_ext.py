from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import Regularizer


class WeightRegularizerWithPmask(Regularizer):
    def __init__(self, l1=0., l2=0.,p_mask=None):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.p_mask = p_mask
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss
        p = self.p
        if self.p_mask is not None:
            p = self.p * K.cast_to_floatx(self.p_mask)
        if self.l1:
            regularized_loss += K.mean(K.abs(p)) * self.l1
        if self.l2:
            regularized_loss += K.mean(K.square(p)) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}