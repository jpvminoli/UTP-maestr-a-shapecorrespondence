clear;clc;close all;

path=pwd;
addpath(genpath(path));

name='2-Left-Cerebral-White-Matter';
load(['mv-1-',name])
shape1=surface; 
Y1=totalSIHKS(shape1);

load(['mv-3-',name])
shape2=surface; 
Y2=totalSIHKS(shape2);

Y=[Y1;Y2];

% Set up model
options = vargplvmOptions('dtcvar');
options.kern = {'rbfard2', 'white'};
options.numActive = 100; 
options.scale2var1 = 1; % scale data to have variance 1
options.optimiser = 'scg';
latentDim = 3;
d = size(Y, 2);

% create the model
gpmodel = vargplvmCreate(latentDim, d, Y, options);
%
gpmodel = vargplvmParamInit(gpmodel, gpmodel.m, gpmodel.X); 

% Optimise the model.
iters = 100;
display = 1;

gpmodel = vargplvmOptimise(gpmodel, display, iters);
%[X, varx_star_all, mini]=vargplvmPredictLatent(model, Y,1:size(Y,1),true,0);
X=gpmodel.X;
[y, model, L] = mixGaussVb(X',4);
save(['gp-vgmm-',name],'model','gpmodel')

figure;
subplot(1,2,1)
plotMesh(shape1,y')
subplot(1,2,2)
%plotMesh(shape2,y(size(Y1,1)+1:size(Y1,1)+size(Y2,1))')

scatter(X(:,1),X(:,2),1,y')