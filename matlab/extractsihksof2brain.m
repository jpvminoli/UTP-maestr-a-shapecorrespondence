% sano-1-2-Left-Cerebral-White-Matter

clear;clc;close all;

path=pwd;
addpath(genpath(path));
basesave='D:\Maestria\shapecorrespondence\data\curvature\';
baseshapes='D:\Maestria\shapecorrespondence\data\brain\';

structure='-2-Left-Cerebral-White-Matter';%'-2-Left-Cerebral-White-Matter' -43-Right-Lateral-Ventricle;
surface1=['sano-1',structure];
surface2=['mv-1',structure];
resample=5000;

descriptortype='SIHKS';

load(surface1)
shape1=surface;
shape1=resampleIf(shape1,resample);
surface=shape1;
% save([baseshapes,surface1,'.mat'],'surface')
% X1=spectraldescriptors(shape1,descriptortype);

load(surface2)
shape2=surface;
shape2=resampleIf(shape2,resample);
surface=shape2;
% save([baseshapes,surface2,'.mat'],'surface')
% X2=spectraldescriptors(shape2,descriptortype);

X=[X1;X2];
% save([basesave,surface1,'-',surface2,'.mat'],'X')