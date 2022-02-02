%% Ejemplo Cambio coordenadas
% 1. Translate the middle of the blob to the origin.
% 
% 2. Rotate the blob.
% 
% 3. Translate the rotated blob back to its starting location.

function [volBrainTr]=transformAff3DVol(volBrain,meta)
% blob_center=mean(volBrain,1);
ori=meta.spaceorigin;
ori=str2num(ori(2:end-1));
ori=zeros(1,3);
blob=volBrain;
blob_center = (size(volBrain) + 1) / 2; %calcula el centro
% Primera matriz de transforamcion
T1 = [1 0 0 0;0 1 0 0;0 0 1 0;-blob_center-ori 1];
%Now here's the rotation.
thetax = 0;
thetay = pi/2;
thetaz = pi/2;
% trans='x';
% switch trans
%     case 'z'
% Tz
T2z = [cos(thetaz)  -sin(thetaz) 0  0;...
    sin(thetaz)    cos(thetaz)  0   0;...
    0    0       1   0;...
    0    0      0     1];
%     case 'y'
T2y = [cos(thetay)  0      -sin(thetay)   0;...
    0             1              0     0;...
    sin(thetay)    0       cos(thetay)   0;...
    0             0              0     1];
%     case 'x'
T2x = [1  0     0    0;...
    0             cos(thetax)              -sin(thetax)     0;...
    0   sin(thetax)       cos(thetax)   0;...
    0             0              0     1];
% end 
% And here's the final translation.
T3 = [1 0 0 0;0 1 0 0;0 0 1 0;blob_center-ori 1];
% The forward mapping is the composition of T1, T2, and T3.
Ta = T1 * T2x * T3; Tb = T1 * T2y * T3; Tc = T1 * T2z * T3;
T=Tb*Tc*Tc*Tc*Tc;

tform = maketform('affine', T);
% Let's do a quick sanity check: the tform struct should map the blob center to itself.
tformfwd(blob_center, tform);
% What the tformarray inputs mean
% Now let's see how to make the inputs to the tformarray function. The syntax of tformarray is B = tformarray(A, T, R, TDIMS_A, TDIMS_B, TSIZE_B, TMAP_B, F).
% A is the input array, and T is the tform struct.

% R is a resampler struct produced by the makeresampler function. You tell makeresampler the type of interpolation you want, as well as how to handle array boundaries.

R = makeresampler('linear', 'fill');
% TDIMS_A specifies how the dimensions of the input array correspond to the dimensions of the spatial transformation represented by the tform struct. Here I'll use the simplest form, in which each spatial transformation dimension corresponds to the same input array dimension. (Don't worry about the details here. One of these days I'll write a blog posting showing an example of when you might want to do something different with this dimension mapping.)

TDIMS_A = [1 2 3];
% TDIMS_B specifies how the dimensions of the output array correspond to the dimensions of the spatial transformation.

TDIMS_B = [1 2 3];
% TSIZE_B is the size of the output array.

TSIZE_B = size(blob);
% TMAP_B is unused when you have a tform struct. Just specify it to be empty.

TMAP_B = [];
% F specifies the values to use outside the boundaries of the input array.

F = 0;
% Call tformarray to transform the blob
blob2 = tformarray(blob, tform, R, TDIMS_A, TDIMS_B, TSIZE_B, TMAP_B, F);
% Display the rotated blob

% clf
% p = patch(isosurface(blob2,0.5));
% set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
% daspect([1 1 1]);
% view(3)
% camlight
% lighting gouraud
volBrainTr=blob2;