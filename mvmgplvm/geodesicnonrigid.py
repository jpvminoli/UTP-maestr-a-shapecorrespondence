import numpy as np
import matplotlib.pyplot as plt
import gpflowmodi.volumes.strmesh as vol
import scipy.io as sio
import open3d as o3d
import os,fnmatch
import gdist

gtfilename='data/gt/'
shapes=['cat','centaur','david','dog','horse','michael','victoria','wolf']
base='data/nonrigid3d/'
gpmethod='mvmgplvm'

for group in shapes:
    print('----Este es el grupo:',group,'---')
    surface1=group+'0'

    data=surface1+'.mat'
    mesh1=vol.StrMesh(filename=base+data)

    files=os.listdir(base)
    pattern=group+'*.mat'
    for fi in files:
        if fnmatch.fnmatch(fi, pattern):
            surface2=fi.split('.')[0]
            if surface1!=surface2:
                print('----Superficie:',surface2,'----')
                try:
                    filename=gtfilename+surface1+'-'+surface2+'.mat'
                    info=sio.whosmat(filename)[0]
                    mat=sio.loadmat(filename)
                    gt=mat[info[0]]-1

                    data=surface2+'.mat'
                    mesh2=vol.StrMesh(filename=base+data)

                    loadruta='results/'+gpmethod+'/labels-nonrigid/'+surface1+'-'+surface2+'.txt'
                    Z=np.loadtxt(loadruta, dtype=float)

                    tam1=mesh1.vertices.shape[0]
                    tam2=mesh2.vertices.shape[0]
                    Z1=Z[0:tam1]
                    Z2=Z[tam1:]

                    norm=np.sqrt(mesh2.getArea())

                    vertices=mesh2.vertices.astype(np.float64)
                    triangles = mesh2.triangles.astype(np.int32)
                    geodist=np.array([])
                    for i in range(tam1):
                        if not(np.isnan(gt[i])):
                            if (i%10)==0:
                                print('Vertex number:',i)
                            label=Z1[i]
                            src=np.array(gt[i],dtype=np.int32)
                            trg=np.where(Z2==label)[0].astype(np.int32)
                            dist=gdist.compute_gdist(vertices, triangles, source_indices=src, target_indices=trg)
                            geodist=np.concatenate((geodist,dist.min(keepdims=True)))#dist.min(keepdims=True)
                    geodist=geodist/norm

                    print('---Finaliza distancia geodesica---')
                    saveruta='results/'+gpmethod+'/geodesicdistance-nonrigid/'+surface1+'-'+surface2+'.txt'

                    np.savetxt(saveruta, geodist, fmt='%f')
                except:
                    print('Ha surgido un error en '+surface1+'-'+surface2)
        



