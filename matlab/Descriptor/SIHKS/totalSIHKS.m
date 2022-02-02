function [X]=totalSIHKS(shape)
    
    K = 100;            % number of eigenfunctions
    alpha = 2;          % log scalespace basis
    T = 1:0.2:20;    % time scales for SI-HKS
    Omega = 2:20;       % frequencies for SI-HKS

    [shape.W, shape.A] = laplaceOperator(shape);
    shape.A = spdiags(shape.A,0,size(shape.A,1),size(shape.A,1));

    % compute eigenvectors/values
    [shape.evecs,shape.evals] = eigs(shape.W,shape.A,K,'SM');
    shape.evals = -diag(shape.evals);

    % compute descriptors
    [shape.sihks, shape.schks] = sihks(shape.evecs,shape.evals,alpha,T,Omega); 

    X=shape.sihks;
end