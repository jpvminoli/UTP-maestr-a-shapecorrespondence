function descriptors=spectraldescriptors(shape,DescriptorType)
    K=150;
    %DescriptorType =  GPS HKS WKS SIHKS HMS SGWS
    [shape.W, shape.A] = laplaceOperator(shape);
    shape.A = spdiags(shape.A,0,size(shape.A,1),size(shape.A,1));

    % compute eigenvectors/values
    [shape.evecs,shape.evals] = eigs(shape.W,shape.A,K,'SM');
    shape.evals = -diag(shape.evals);
    
    descriptors = Get_Signature(shape.evecs, shape.evals,DescriptorType);
end
