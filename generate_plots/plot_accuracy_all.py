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
#l1 = float(sys.argv[2])
#layer=int(sys.argv[3])

l1_set = [0.0,0.5]
layer_set = [1,3]

if dataset =='MSR3D':
    ylim = [0.8,1]
    layer_set = [1]
    action_set =['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch',
'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',
'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', 
'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw' ]
elif dataset =='MSRDaily':
    ylim = [0.5,0.85]
    action_set = ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop',
    'use vacuum', 'cheer up', 'sit still', 'toss paper', 'play game', 'lay down on sofa',
    'walk', 'play guitar', 'stand up', 'sit down']

elif dataset =='CompAct':
    ylim = [0.8,1]
    action_set=['composed 1','composed 2','composed 3','composed 4',
    'composed 5','composed 6','composed 7','composed 8',
    'hand wave+drink','talk phone+drink','talk phone+pick up','talk phone+scratch head',
    'walk+calling','walk+clapping', 'walk+hand waving','walk+reading']

fstr='dict_size/nword{}_lr0.05_objhinge_opt0_decay50_l1{}_reg0.001_layer{}_rs{}_multi2.mat'

# l1 o-
# l2 D
# layer 1 m
# layer 3 b
# multiscale .-
pattern = ['yd-','bd--','m*-','g*--']
acc_all = []#p.zeros(6)
nword_set = [5,20,40,60,80,100] 
legend = []
idx =0
for layer in layer_set:
    for l1 in l1_set:
        acc = np.zeros(len(nword_set))
    
        for i,nword in enumerate(nword_set):
            tmp = np.zeros(10)
            for rs in range(10):
                fname=fstr.format(nword,l1,layer,rs)
                if os.path.isfile(fname):
                    print fname
                    a=sio.loadmat(fname)
                    tmp[rs] = np.mean(a['test_acc_all'])
            acc[i]= tmp[tmp>0].mean()
            print(acc)
        acc_all += [acc]
        # add plot and legend here
        plt.plot(acc,pattern[idx],linewidth=3,markersize=8, label='TP{}_alpha{}'.format(layer,l1))
        idx +=1

fstr='multiscale/nword{}_lr0.05_objhinge_opt0_decay50_l1{}_reg0.001_layer{}_rs{}_multi1.mat'
l1=0.0
idx = 0
pattern = ['cd-.','r*-.']
for layer in layer_set:
    acc = np.zeros(len(nword_set))

    for i,nword in enumerate(nword_set):
        tmp = np.zeros(10)
        for rs in range(10):
            fname=fstr.format(nword,l1,layer,rs)
            if os.path.isfile(fname):
                print fname
                a=sio.loadmat(fname)
                tmp[rs] = np.mean(a['test_acc_all'])
        acc[i]= tmp[tmp>0].mean()
        print acc
    acc_all += [acc]
    # add plot and legend here
    plt.plot(acc,pattern[idx],linewidth=3,markersize=8, label='TP{}_alpha{}_multiscale'.format(layer,l1))
    idx +=1



print acc_all
ax = plt.gca()
ax.set_ylim(ylim)
plt.xlabel('MP_per_BP',fontsize=18)
plt.ylabel('accuracy',fontsize=18)
plt.xticks(range(6),['5','20','40','60','80','100'])
plt.legend(loc='lower right')
plt.title('Performance on {}'.format(dataset),fontsize=24)
#plt.title('{}_accuracy_for_different_dictsize_l1{}_layer{}'.format(dataset,nword,layer))
#os.mkdir('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/{}'.format(dataset))
plt.savefig('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/{}_performance.png'.format(dataset))        
