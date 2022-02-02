clear;clc
load FreeSurferTable.mat;

init=168;
fin=init+1;
% val=[3201,3203,3204,3205,3206,3207,4201,4203,4204,4205,4206,4207];
% color=[235,35,95;35,75,35;135,55,95;115,35,35;35,195,35;20,220,160;235,35,95;35,75,35;135,155,195;115,35,35;35,195,35;20,220,160];
% name={'wm-lh-frontal-lobe','wm-lh-cingulate-lobe','wm-lh-occiptal-lobe','wm-lh-temporal-lobe','wm-lh-parietal-lobe','wm-lh-insula-lobe','wm-rh-frontal-lobe','wm-rh-cingulate-lobe','wm-rh-occiptal-lobe','wm-rh-temporal-lobe','wm-rh-parietal-lobe','wm-rh-insula-lobe'};
% val=[170,171,172,173,174,175];
% color=[119,159,176;119,0,176;119,100,176;242,104,76;206,195,58;119,159,176];
% name={'brainstem','DCG','Vermis','Midbrain','Pons','Medulla'};
val=[192];
color=[250,255,50];
name={'Corpus_Callosum'};

infoL_FSstruct.labels=[infoL_FSstruct.labels(1:init),val,infoL_FSstruct.labels(fin:end)];
infoL_FSstruct.RGBcolor=[infoL_FSstruct.RGBcolor(1:init,:);color;infoL_FSstruct.RGBcolor(fin:end,:)];


initSize=length(infoL_FS);
addSize=length(val);

infoFS=infoL_FS(1:init,:);
infonames=infoL_FSstruct.structuresNames(1:init);
for i=init+1:init+addSize
    infoFS(i,:)={val(i-init),name{i-init},color(i-init,1),color(i-init,2),color(i-init,3),0};
    infonames(i)=name(i-init);
end
infoFS(init+addSize+1:initSize+addSize,:)=infoL_FS(fin:end,:);
infonames(init+addSize+1:initSize+addSize)=infoL_FSstruct.structuresNames(fin:end);
infoL_FS=infoFS;
infoL_FSstruct.structuresNames=infonames;

perinatalLabels=[2,3,7,8,9,10,11,12,13,16,41,42,46,47,48,49,50,51,52,136,137,169,170,172,173,174,175,176,192,1022,1024,2022,2024,3201,3203,3204,3205,3206,3207,4201,4203,4204,4205,4206,4207];
save('FreeSurferTable','perinatalLabels','infoL_FS','infoL_FSstruct','DataLabelFS');