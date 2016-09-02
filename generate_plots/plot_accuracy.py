'''
For each dataset, plot # activated mp per part per action for each alpha
'''


import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
dataset = sys.argv[1]
nword = int(sys.argv[2])

layer=1
if dataset =='MSR3D':
    action_set =['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch',
'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',
'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', 
'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw' ]
elif dataset =='MSRDaily':
    layer = 3
    action_set = ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop',
    'use vacuum', 'cheer up', 'sit still', 'toss paper', 'play game', 'lay down on sofa',
    'walk', 'play guitar', 'stand up', 'sit down']

elif dataset =='CompAct':
    action_set=['composed 1','composed 2','composed 3','composed 4',
    'composed 5','composed 6','composed 7','composed 8',
    'hand wave+drink','talk phone+drink','talk phone+pick up','talk phone+scratch head',
    'walk+calling','walk+clapping', 'walk+hand waving','walk+reading']

fstr='nword{}_lr0.05_objhinge_opt0_decay50_l1{}_reg0.001_layer{}_rs0_multi2.mat'


acc = np.zeros(11)
for l1 in range(11):
    fname=fstr.format(nword,0.1*l1,layer)
    if os.path.isfile(fname):
        a=sio.loadmat(fname)
        
        cv_acc=[]
        for idx in range(len(a['hist_all'][0])):
            cv_acc += [a['hist_all'][0][idx][0][0][2][-1][-1]]
        acc[l1] = np.mean(cv_acc)  
acc = acc[acc>0]
        
plt.plot(acc)
plt.title('{}_nword{}'.format(dataset,nword))
plt.savefig('plots/accurary_{}_nword{}.png'.format(dataset,nword))        
