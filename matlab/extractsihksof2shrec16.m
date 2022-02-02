clear;clc;close all;

path=pwd;
addpath(genpath(path));
basesave='D:\Maestria\shapecorrespondence\data\curvature\shrec16/';
baseshapes='D:\Maestria\shapecorrespondence\data\shrec2016\';
basegt='D:\Maestria\shapecorrespondence\data\gt\shrec16\';
model='cat';
type='holes';
shape='1';
descriptortype='SIHKS';

N = load_off([type '_' model '_shape_' shape '.off']);
N.X=N.VERT(:,1);
N.Y=N.VERT(:,2);
N.Z=N.VERT(:,3);
N.baryc_gt = load([type '_' model '_shape_' shape '.baryc_gt']);
surface=N;
save([baseshapes,type,'\',type '_' model '_shape_' shape,'.mat'],'surface')
X1=spectraldescriptors(surface,descriptortype);

[~,I]=max(N.baryc_gt(:,2:4),[],2);
trian=N.baryc_gt(:,1);

M = load_off([model '.off']);
M.X=M.VERT(:,1);
M.Y=M.VERT(:,2);
M.Z=M.VERT(:,3);
surface=M;
save([baseshapes,'null\',model,'.mat'],'surface')
X2=spectraldescriptors(surface,descriptortype);

gt2=diag(M.TRIV(trian,I));
X=[X1;X2];

T12gt=gt2;
save([basesave,model,'-',type,'-',shape,'.mat'],'X')
save([basegt,model,'-',type,'-',shape,'.mat'],'T12gt')