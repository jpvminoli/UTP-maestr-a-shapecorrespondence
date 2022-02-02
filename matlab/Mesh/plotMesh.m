function []=plotMesh(surface,color)
    if(nargin<2)
        trisurf(surface.TRIV,surface.X,surface.Y,surface.Z), axis image, shading interp, 
        view([-10 20]),
        title('Surface Graph')
        axis off
        camlight
        colormap([1 1 1])
    else
        tam=size(color);
        if(tam(1)>2)
            trisurf(surface.TRIV,surface.X,surface.Y,surface.Z,color), axis image, shading interp, 
            view([-10 20]),
            title('Surface Graph')
            axis off
            camlight
        else
            trisurf(surface.TRIV,surface.X,surface.Y,surface.Z), axis image, shading interp, 
            view([-10 20]),
            title('Surface Graph')
            axis off
            camlight
            colormap(color)
        end
        
    end
end