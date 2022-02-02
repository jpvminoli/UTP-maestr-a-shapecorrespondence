function [W, A] = laplaceOperator(shape)
    if nargin < 1
        error('Too few input arguments');
    end
    opt.hs = 2;
    opt.rho = 3;
    opt.htype = 'ddr';
    opt.dtype = 'cotangent';
    
    [II, JJ, SS, AA] = meshlp(shape.TRIV, shape.X, shape.Y, shape.Z, opt);
    W = sparse(II, JJ, SS);
    A = AA;
end