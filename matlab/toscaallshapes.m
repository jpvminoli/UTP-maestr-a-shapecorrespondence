%big tosca

clear;clc;close all;

path=pwd;
addpath(genpath(path));
basesave='D:\Maestria\shapecorrespondence\data\curvature\tosca\';
baseshapes='D:\Maestria\shapecorrespondence\data\tosca\';
basegt='D:\Maestria\shapecorrespondence\data\gt\tosca\';

shapes=["cat","centaur","david","dog","horse","michael","victoria","wolf"];
descriptortype='SIHKS';
datasetruta='Data\toscahires-mat\';


for group=shapes
    forma=convertStringsToChars(group);
    disp(forma)
    surface1=[forma,'0'];
    
    load(surface1)
    save([baseshapes,surface1,'.mat'],'surface')
    X1=spectraldescriptors(surface,descriptortype);
    disp(size(X1))
    files = dir([datasetruta,forma,'*.mat']);
    
    for fi=1:numel(files)
        fname = [files(fi).name];
        [~, part_name, ~] = fileparts(fname);
        surface2=part_name;
        if ~strcmp(surface1,surface2)
            disp(surface2)
            load(surface2)
            save([baseshapes,surface2,'.mat'],'surface')
            X2=spectraldescriptors(surface,descriptortype);%totalSIHKS(shape2);
            disp(size(X2))
            X=[X1;X2];
            save([basesave,surface1,'-',surface2,'.mat'],'X')
        end
    end 
end
