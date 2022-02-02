clear;clc
load FreeSurferTable.mat;

origSize=length(infoL_FS);
periNames={};
j=1;
for i=1:origSize
    if sum(perinatalLabels==infoL_FS{i,1})==1
        periNames{j}=infoL_FS{i,2};
        j=j+1;
    end
end
