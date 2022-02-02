function shapeTr=resampleIf(shape,limit)

    tam=length(shape.X);
    
    if tam>limit
        ratio=limit/tam;
        shapeTr=resampleMesh(shape,ratio);
    else
        shapeTr=shape;
    end

end