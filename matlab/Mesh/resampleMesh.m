function [surf]=resampleMesh(shape,keepratio)
    shape.vertices=zeros(length(shape.X),3);
    shape.vertices(:,1)=shape.X;
    shape.vertices(:,2)=shape.Y;
    shape.vertices(:,3)=shape.Z;

    [surf.vertices,surf.TRIV]=meshresample(shape.vertices,shape.TRIV,keepratio);

    surf.X=surf.vertices(:,1);
    surf.Y=surf.vertices(:,2);
    surf.Z=surf.vertices(:,3);
end