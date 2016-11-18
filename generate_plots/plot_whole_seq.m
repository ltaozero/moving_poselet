addpath(genpath('~/Desktop/lab/matlab_toolbox/mocgui-master/'));
load('~/work/Data/MSR3D/MSR3D_Feature_norm.mat','label','subject','features');

config = cell(1,10);
config{1} = [4:7];
config{2} = [5,14,16,18];
config{3} = [6,15,17,19];
config{4} = [7,5,6,14:19];
config{5} = [3,1,8,10,12];
config{6} = [3,2,9,11,13];
config{7} = [20,1:4,8:13];
config{8} = [20,1:7];
config{9} = [20,1:13];
config{10} = [1:20];
config = config(~cellfun('isempty',config));
Conn = [20,3;1,3;2,3;1,8;8,10;10,12;2,9;9,11;11,13;3,4;...
     4,7;7,5;7,6;5,14;6,15;14,16;16,18;15,17;17,19]';

actions = {'high arm wave', 'horizontal arm wave', 'hammer', 'hand catch',...
'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',...
'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', ...
'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw'  };

sample_idx = 226
feat = features{sample_idx}
T = size(feat,2)

 
feat = feat(:,10:3:end-10)
T = size(feat,2)
feat = reshape(feat,3,20,T); 
% add shift
for i = [1:T];
    feat(3,:,i) = feat(3,:,i)+ 0.2*i
    feat(1,:,i) = feat(1,:,i)- 0.4*i
end
    
anMocap_easy_img(feat, Conn,[],...
sprintf('MP_img/action%s_sample%d', actions{label(sample_idx)},sample_idx),actions{label(sample_idx)});
