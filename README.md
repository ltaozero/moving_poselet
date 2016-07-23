# moving_poselet
This code implements the Moving Poselet model described in the following paper:

L. Tao and R. Vidal. 
Moving Poselets: A Discriminative and Interpretable Skeletal Motion Representation for Action Recognition.
In ChaLearn Looking at People Workshop 2015, 2015.

To run the experiments, install [Keras 1.0 version] (http://keras.io/) following their instructions [here] (http://keras.io/#installation) and run the following command:

Use CPU:
```
python moving_poselet_exp.py DATASET NB_MOVING_POSELET_PER_PART TEMPORAL_PYRAMID_LAYER
```
e.g. 
```
python moving_poselet_exp.py MSRDaily 50 3
python moving_poselet_exp.py MSR3D 50 1
```
Use GPU
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python moving_poselet_exp.py MSRDaily 50 3
```

To see other parameter options, run
```
python moving_poselet_exp.py -h 
```
