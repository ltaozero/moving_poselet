'''
For each dataset, plot # activated mp per part per action for each alpha
'''


import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
dataset = sys.argv[1]
l1 = float(sys.argv[2])

layer=int(sys.argv[3])
if dataset =='MSR3D':
    action_set =['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch',
'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',
'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', 
'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw' ]
elif dataset =='MSRDaily':
    action_set = ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop',
    'use vacuum', 'cheer up', 'sit still', 'toss paper', 'play game', 'lay down on sofa',
    'walk', 'play guitar', 'stand up', 'sit down']

elif dataset =='CompAct':
    action_set=['composed 1','composed 2','composed 3','composed 4',
    'composed 5','composed 6','composed 7','composed 8',
    'hand wave+drink','talk phone+drink','talk phone+pick up','talk phone+scratch head',
    'walk+calling','walk+clapping', 'walk+hand waving','walk+reading']

fstr='nword{}_lr0.05_objhinge_opt0_decay50_l1{}_reg0.001_layer{}_rs{}_multi2.mat'


acc_all = np.zeros(6)
acc = np.zeros((6,10))
nword_set = [5,20,40,60,80,100] 
for i,nword in enumerate(nword_set):
    for rs in range(10):
        fname=fstr.format(nword,l1,layer,rs)
        
        if os.path.isfile(fname):
            a=sio.loadmat(fname)
                
            acc[i,rs] = np.mean(a['test_acc_all'])
    acc_all[i] = np.mean(acc[i][acc[i]>0])        

print acc
plt.plot(acc_all)
plt.title('{}_accuracy_for_different_dictsize_l1{}_layer{}'.format(dataset,nword,layer))
#os.mkdir('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/{}'.format(dataset))
plt.savefig('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/{}/accurary_for_different_dictsize_l1{}_layer{}.png'.format(dataset,l1,layer))        
