clear;clc;close all;

path=pwd;
addpath(genpath(path));
basesave='D:\Maestria\shapecorrespondence\data\curvature\shrec16\';
baseshapes='D:\Maestria\shapecorrespondence\data\shrec2016\';
basegt='D:\Maestria\shapecorrespondence\data\gt\shrec16\';

type='cuts';
descriptortype='SIHKS';

datasetruta='Data\shrec2016\null\';
if strcmp(type,'holes')
    datastype='Data\shrec2016\holes\';
else
    datastype='Data\shrec2016\cuts\';
end


filesmodel = dir([datasetruta,'*.off']);

for fimodel=1:numel(filesmodel)
    fname = [filesmodel(fimodel).name];
    [~, part_name, ~] = fileparts(fname);
    model=part_name;
    disp('Inicio Modelo:')
    disp(model)
    
    M = load_off([model '.off']);
    M.X=M.VERT(:,1);
    M.Y=M.VERT(:,2);
    M.Z=M.VERT(:,3);
    surface=M;
    save([baseshapes,'null\',model,'.mat'],'surface')
    X2=spectraldescriptors(surface,descriptortype);
    
    filesshape = dir([datastype,type,'_',model,'*.off']);
    for fishape=1:numel(filesshape)
        fname = [filesshape(fishape).name];
        [~, part_name, ~] = fileparts(fname);
        shape=part_name;
        
        N = load_off([shape '.off']);
        N.X=N.VERT(:,1);
        N.Y=N.VERT(:,2);
        N.Z=N.VERT(:,3);
        N.baryc_gt = load([shape '.baryc_gt']);
        surface=N;
        save([baseshapes,type,'\',shape,'.mat'],'surface')
        X1=spectraldescriptors(surface,descriptortype);
        
        X=[X1;X2];
        
        save([basesave,type,'\',shape,'.mat'],'X')
        
        [~,I]=max(N.baryc_gt(:,2:4),[],2);
        trian=N.baryc_gt(:,1);
        
        gt2=diag(M.TRIV(trian,I));
        
        T12gt=gt2;
        save([basegt,type,'\',shape,'.mat'],'T12gt')
    end
end
