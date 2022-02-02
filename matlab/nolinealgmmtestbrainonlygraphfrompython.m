clear;clc;close all;

path=pwd;
addpath(genpath(path));

structure='-43-Right-Lateral-Ventricle';
surface1=['sano-1',structure];
surface2=['mv-1',structure];
resample=5000;

namelabel=['label-',surface1,'-',surface2];


load(surface1)
shape1=surface;
shape1=resampleIf(shape1,resample);
surface=shape1;

load(surface2)
shape2=surface;
shape2=resampleIf(shape2,resample);
surface=shape2;

load(namelabel)
subplot(1,2,1)
plotMesh(shape1,Z1')
title('')%healthy
subplot(1,2,2)
plotMesh(shape2,Z2')
title('') %sick