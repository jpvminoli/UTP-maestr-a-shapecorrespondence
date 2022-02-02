%low tosca
clear;clc;close all;

path=pwd;
addpath(genpath(path));
basesave='D:\Maestria\shapecorrespondence\data\curvature\';
baseshapes='D:\Maestria\shapecorrespondence\data\nonrigid3d\';
basegt='D:\Maestria\shapecorrespondence\data\gt\';
shapes=["cat","centaur","david","dog","horse","michael","victoria","wolf"];
descriptortype='SIHKS';
datasetruta='Dataset\TOSCA_Isometric\vtx_5k\';

for group=shapes
    forma=convertStringsToChars(group);
    disp(forma)
    surface1=[forma,'0'];
    
    files = dir([datasetruta,forma,'*.off']);
 
    for fi=1:numel(files)
        fname = [files(fi).name];
        [~, part_name, ~] = fileparts(fname);
        surface2=part_name;
        
        S1 = MESH.MESH_IO.read_shape([surface1,'.off']);
        gt1 = FILE_IO.read_landmarks([surface1,'.sym.vts']);

        shape1=S1.surface;
        surface=shape1;
        save([baseshapes,surface1,'.mat'],'surface')
        X1=spectraldescriptors(shape1,descriptortype);%totalSIHKS(shape1);

        S2 = MESH.MESH_IO.read_shape([surface2,'.off']);
        gt2 = FILE_IO.read_landmarks([surface2,'.sym.vts']);

        shape2=S2.surface;
        surface=shape2;
        save([baseshapes,surface2,'.mat'],'surface')
        X2=spectraldescriptors(shape2,descriptortype);%totalSIHKS(shape2);

        X=[X1;X2];

        T12gt = nan(S1.nv,1); T12gt(gt1) = gt2;
        save([basegt,surface1,'-',surface2,'.mat'],'T12gt')
        save([basesave,surface1,'-',surface2,'.mat'],'X')
    end 
end

