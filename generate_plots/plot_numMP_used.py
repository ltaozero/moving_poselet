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

fig, axes = plt.subplots(3,4,figsize=(40,20))
for l1 in range(11):
    ax = axes[l1//4][l1%4]
    fname=fstr.format(nword,0.1*l1,layer)
    if os.path.isfile(fname):
        a=sio.loadmat(fname)
        MP_used = np.zeros((len(action_set),10))
        for action in range(len(action_set)):
            weights = abs(a['weights_all'][0,2][:,action])
            weights = np.reshape(weights,(np.power(2,layer)-1,-1)).sum(axis=0)
            maxv = max(weights)
            num_MP = nword
            for part in range(10):
                MP_used[action,part] = (weights[num_MP*part:num_MP*(part+1)]> 0.05*maxv).sum()
        im=ax.imshow(MP_used,interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        ax.set_aspect('auto')
        ax.set_title('alpha ={}'.format(0.1*l1))

        tickpos = np.arange(10)
        ticklabel = ['B','LL','RL','LB','LA','RA','UB','T','FUB','FB']
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticklabel)
        ax.set_yticks(np.arange(len(action_set)))
        ax.set_yticklabels(action_set)
        
fig.suptitle('{}_nword{}'.format(dataset,nword))
fig.savefig('plots/numMP_used_{}_nword{}.png'.format(dataset,nword, action_set[action]))        



