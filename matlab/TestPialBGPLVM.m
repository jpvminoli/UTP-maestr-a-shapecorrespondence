clear;clc;close all;

path=pwd;
addpath(genpath(path));

name='jgh';
resonances={'1','2'};
Y=[];
shape={};
i=1;
for res=resonances
    resonance=res{1};
    load([name,'-',resonance,'-pial.mat'])
    shape{i}=surface;
%     shape{i}=resampleIf(shape{i},15000);
    Y=[Y;totalSIHKS(shape{i})];
    i=i+1;
end
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
iters = 100;
display = 1;

gpmodel = vargplvmOptimise(gpmodel, display, iters);
X=gpmodel.X;
[y, model, L] = mixGaussVb(X',5);
y=y';
tam1=length(shape{1}.X);
out{1}=y(1:tam1,:);
out{2}=y(tam1+1:end,:);

for i=1:2
    subplot(2,2,i)
    plotMesh(shape{i},out{i})
end
subplot(2,2,[3,4])
scatter(X(:,1),X(:,2),1,y)
