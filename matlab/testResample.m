clear;clc;close all;

path=pwd;
addpath(genpath(path));

load mv-1-2-Left-Cerebral-White-Matter
shape1=surface;
shape1=resampleIf(shape1,3600);
Y=totalSIHKS(shape1);

% Set up model
options = vargplvmOptions('dtcvar');
options.kern = {'rbfard2', 'white'};
options.numActive = 100; 
options.scale2var1 = 1; % scale data to have variance 1
%options.tieParam = 'tied';  

options.optimiser = 'scg';
latentDim = 2;
d = size(Y, 2);

% create the model
gpmodel = vargplvmCreate(latentDim, d, Y, options);
%
gpmodel = vargplvmParamInit(gpmodel, gpmodel.m, gpmodel.X); 

% Optimise the model.
iters = 1000;
display = 1;

gpmodel = vargplvmOptimise(gpmodel, display, iters);
X=gpmodel.X;
[y, model, L] = mixGaussVb(X',5);

subplot(1,2,1)
plotMesh(shape1,y')
subplot(1,2,2)
scatter(X(:,1),X(:,2),1,y')

%save('latentgp-vgmm-mv1-2-v2','X','y','shape1','model','gpmodel')