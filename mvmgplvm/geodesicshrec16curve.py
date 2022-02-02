import numpy as np
import matplotlib.pyplot as plt
import gpflowmodi.volumes.strmesh as vol
import scipy.io as sio
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn import mixture
import open3d as o3d
import os,fnmatch
import gdist


tipo='cuts'
xfilename='data/curvature/shrec16/'+tipo+'/'
gtfilename='data/gt/shrec16/'+tipo+'/'
gpmethod='mvmgplvm'
base='data/shrec2016/'

filesmodel=os.listdir(base+'null/')
pattern='*.mat'
for fimodel in filesmodel:
    if fnmatch.fnmatch(fimodel, pattern):
        model=fimodel.split('.')[0]
        print('----Este es el grupo:',model,'---')
        data=model+'.mat'
        mesh2=vol.StrMesh(filename=base+'null/'+data)

        filesshape=os.listdir(base+tipo+'/')
        pattern2=tipo+'_'+model+'*.mat'
        for fishape in filesshape:
            if fnmatch.fnmatch(fishape, pattern2):
                shape=fishape.split('.')[0]
                data=shape+'.mat'
                print('----Forma:',shape,'----')
                try:
                    mesh1=vol.StrMesh(filename=base+tipo+'/'+data)
                    
                    filename=gtfilename+shape+'.mat'
                    info=sio.whosmat(filename)[0]
                    mat=sio.loadmat(filename)
                    gt=mat[info[0]]-1
                    tamgt=gt.shape[0]

                    loadruta='results/'+gpmethod+'/labels-shrec/'+tipo+'/'+shape+'.txt'
                    Z=np.loadtxt(loadruta, dtype=float)

                    tam1=mesh1.vertices.shape[0]
                    tam2=mesh2.vertices.shape[0]
                    Z1=Z[0:tam1]
                    Z2=Z[tam1:]

                    norm=np.sqrt(mesh1.getArea())

                    vertices=mesh1.vertices.astype(np.float64)
                    triangles = mesh1.triangles.astype(np.int32)
                    geodist=np.array([])
                    for i in range(tamgt):
                        if not(np.isnan(gt[i])):
                            if (i%10)==0:
                                print('Vertex number:',i)
                            label=Z2[gt[i]]
                            src=np.array([i],dtype=np.int32)
                            trg=np.where(Z1==label)[0].astype(np.int32)
                            dist=gdist.compute_gdist(vertices, triangles, source_indices=src, target_indices=trg)
                            geodist=np.concatenate((geodist,dist.min(keepdims=True)))#dist.min(keepdims=True)
                    geodist=geodist/norm

                    print('---Finaliza distancia geodesica---')
                    saveruta='results/'+gpmethod+'/geodesicdistance-partial/'+tipo+'/'+shape+'.txt'

                    np.savetxt(saveruta, geodist, fmt='%f')
                except:
                    print('Ha surgido un error en '+shape)

