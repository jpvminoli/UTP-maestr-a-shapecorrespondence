function [labels, RGBv, names,peLabels]=findLabelColorFS_all(slice)

load FreeSurferTable.mat;

labelsMaps=unique(slice(:));
commonLabFSySeg=intersect(labelsMaps,infoL_FSstruct.labels);

labels=zeros(length(commonLabFSySeg),1);
RGBv=zeros(length(commonLabFSySeg),3);
names={};
for i=1:length(commonLabFSySeg)
    [~,pos]=find(infoL_FSstruct.labels==commonLabFSySeg(i));
    labels(i)=infoL_FSstruct.labels(pos);
    RGBv(i,:)=infoL_FSstruct.RGBcolor(pos,:);
    names{i}=infoL_FSstruct.structuresNames{pos};
end
peLabels=perinatalLabels;
