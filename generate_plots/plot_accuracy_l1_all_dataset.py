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
#dataset = sys.argv[1]
#nword = int(sys.argv[2])
#layer=int(sys.argv[3])
layer = 1
nword = 100

patterns = ['yd-','bp-.','m*--']
for i,dataset in enumerate(['MSR3D','MSRDaily','CompAct']):
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
        layer = 3
        action_set=['composed 1','composed 2','composed 3','composed 4',
        'composed 5','composed 6','composed 7','composed 8',
        'hand wave+drink','talk phone+drink','talk phone+pick up','talk phone+scratch head',
        'walk+calling','walk+clapping', 'walk+hand waving','walk+reading']

    fstr='{}/elasticnet/nword{}_lr0.05_objhinge_opt0_decay50_l1{}_reg0.001_layer{}_rs{}_multi2.mat'


    acc_all = np.zeros(11)
    acc = np.zeros((11,10))
    for l1 in range(11):
        for rs in range(10):
            fname=fstr.format(dataset,nword,0.1*l1,layer,rs)
            print fname
            if os.path.isfile(fname):
                a=sio.loadmat(fname)
                    
                acc[l1,rs] = np.mean(a['test_acc_all'])
        acc_all[l1] = np.mean(acc[l1][acc[l1]>0])        

    print acc
    plt.plot(acc_all, patterns[i],linewidth=3,markersize=8,label=dataset)
plt.xlabel('alpha')
label = [str(0.1*i) for i in range(11)]
plt.xticks(np.arange(11),label)
plt.ylabel('accuracy')
plt.legend(loc='lower right')
#plt.boxplot(acc.transpose())
#plt.plot(acc_all,marker='D')

ax = plt.gca()
ax.set_ylim([0.6,1])
plt.title('Classification accuracy using different alpha',fontsize=18)
#plt.title('{}:accurary_for_different_l1_nword{}_layer{}'.format(dataset,nword,layer),fontsize=18)
#os.mkdir('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/{}'.format(dataset))
plt.savefig('/home-3/ltao4@jhu.edu/scratch/mp_journal/plots/accurary_for_different_l1_nword{}.png'.format(nword))        
