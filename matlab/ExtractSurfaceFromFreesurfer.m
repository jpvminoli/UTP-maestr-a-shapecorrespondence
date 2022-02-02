clear;clc
close all

path=pwd;
addpath(genpath(path));

rutaMesh='Data\Surface';
pacientName='sano';
resonanceNumber='1';
file='aparcaseg.nrrd';
[slice,meta]=nrrdread([pacientName,'-',resonanceNumber,'-',file]);
[sliceTr]=transformAff3DVol(slice,meta);
slice=sliceTr;
[labels,RGBv,names,peLabels]=findLabelColorFS_all(slice);

for i=1:length(labels)
    %if(labels(i)~=0 && sum(peLabels==labels(i))==1)
        fprintf('Extracting surface number %d from %d...\n',i,length(labels));
        volBrain=isosurface(smooth3(slice==labels(i)));
        if isempty(volBrain.vertices)==0
            ori=meta.spaceorigin;
            ori=abs(str2num(ori(2:end-1))); %#ok<ST2NM>
            tr=[ori(1),ori(2),ori(3)];
            vTr=repmat(tr,size(volBrain.vertices,1),1);
            volBrain.vertices=volBrain.vertices-vTr;
            
            surface.TRIV=volBrain.faces;
            surface.X=volBrain.vertices(:,1);
            surface.Y=volBrain.vertices(:,2);
            surface.Z=volBrain.vertices(:,3);
            
            %graficar
%             plotMesh(surface)

            dMat=[rutaMesh,'\MatlabModels\'];
            dVTK=[rutaMesh,'\VtkModels'];

            name=[pacientName,'-',resonanceNumber,'-',num2str(labels(i)),'-',names{i}];
            exportTriangulation2VTK(name,volBrain.vertices,volBrain.faces,dVTK);
            
            
            save([dMat,name,'.mat'],'surface');
        end
    %end
end