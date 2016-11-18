import argparse
import numpy as np

def mp_parser():
    parser = argparse.ArgumentParser(description='MovingPoselet_parser')

    parser.add_argument('dataset', action="store")
    parser.add_argument('num_MP', action="store", type=int)
    parser.add_argument('tp_layer', action="store", type=int)
    parser.add_argument('--lr', action="store", type=float, dest='learning_rate',default = 0.05)
    parser.add_argument('--l2', action="store", type=float, dest='reg_weight', default = 1e-4)
    parser.add_argument('--opt', action="store", type=int, default = 0,dest='opt_method')
    parser.add_argument('-d', action="store", type=int,default=50, dest='decay')
    parser.add_argument('-w', action="store", type=int,default=5, dest='window_size')
    parser.add_argument('--l1', action="store", type=float, default=0, dest='l1_alpha')
    parser.add_argument('--rs', action="store", type = int, default = 0)
    parser.add_argument('--full', action="store_true", dest='use_fb', default = False)
    parser.add_argument('-s', action="store", type=int, dest='multi_ts', default = 2)
    parser.add_argument('--sub', action="store", type=int, dest='subset',default=1)
    parser.add_argument('-b', action="store", type=int, dest='batch_size', default = 10)
    parser.add_argument('-f', action="store", dest='features', default = 'raw')
    parser.add_argument('--epoch', action="store", type=int, dest='nb_epoch', default = 300)
    parser.add_argument('--exp', action="store", dest='exp_name',default='results')
    return parser

def process_params(params):
    dataset = params['dataset']
    padding = 1

    #features = dataset+'_'+params['featurename']
    features = params['features']
    maxlen_map={'MSR3D':100,'MSRDaily':700,'CompAct':1000,'MHAD':700,'HDM05':1000, 'CAD120':1200,'Suturing':1000, 'KnotTying':1000, 'NeedlePassing':2500}
    maxlen=maxlen_map[params['dataset']]
    #sample_rate_set =[1]
    window_size = params['window_size']
    multi_ts = params['multi_ts']
    if multi_ts ==1:
        sample_rate_set=[1,5]
    elif multi_ts ==2:
        sample_rate_set =[1]
    elif multi_ts ==3:
        sample_rate_set=[2]
    elif multi_ts ==4:
        sample_rate_set=[5]
    elif multi_ts ==0:
        sample_rate_set=[1]
        window_size = 1
    elif multi_ts ==5:
        sample_rate_set = [1,5,10]
    compute_vec = 0
    if window_size > 1:
        compute_vec = 1
    #joint_map={'MSR3D':20,'MSRDaily':20,'CompAct':20,'MHAD':35,'HDM05':31,'CAD120':15}
    joint_map={'MSR3D':20,'MSRDaily':20,'CompAct':20,'MHAD':35,'HDM05':31,'CAD120':15,'Suturing':4, 'KnotTying':4, 'NeedlePassing':4}
    njoints = joint_map[dataset]
    full_BP = np.arange(njoints)+1

    data_gen_params = {'dataset':dataset, 'features':features, 'sample_rate_set':sample_rate_set,'window_size':window_size,'compute_vec': compute_vec, 'padding': padding, 'maxlen': maxlen, 'full_BP': full_BP}
    return data_gen_params
