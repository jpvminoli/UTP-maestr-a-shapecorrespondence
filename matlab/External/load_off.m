% This file implements the method described in:
%
% "Consistent Partial Matching of Shape Collections via Sparse Modeling" 
% L. Cosmo, E. Rodola, A. Albarelli, F. Memoli, D. Cremers
% Computer Graphics Forum, 2016
%
% If you use this code, please cite the paper above.
% 
% Luca Cosmo, Emanuele Rodola (c) 2015

function shape = loadoff(filename)
%% Loads off model

shape = [];

f = fopen(filename, 'rt');

fgetl(f);
n = sscanf(fgetl(f), '%d %d %d');
nv = n(1);
nt = n(2);

data = fscanf(f, '%f');

shape.TRIV = reshape(data(3*nv+1:3*nv+4*nt), [4 nt])';
shape.TRIV = shape.TRIV(:,2:end) + 1;

data = data(1:3*nv);
data = reshape(data, [3 nv]);
% 
% shape.X = data(1,:)';
% shape.Y = data(2,:)';
% shape.Z = data(3,:)';
shape.VERT = data';
fclose(f);
