clear;clc;close all;

path=pwd;
addpath(genpath(path));

name='2-Left-Cerebral-White-Matter';
load(['gp-vgmm-',name])

load(['mv-1-',name])
shape1=surface;
tam1=size(shape1.X,1);

load(['mv-3-',name])
shape2=surface; 
tam2=size(shape2.X,1);

X=gpmodel.X;
[y, R] = mixGaussVbPred(model,X');

figure;
subplot(1,2,1)
plotMesh(shape1,y(1:tam1)')
subplot(1,2,2)
plotMesh(shape2,y(tam1+1:tam1+tam2)')

figure; scatter3(X(:,1),X(:,2),X(:,3),1,y')