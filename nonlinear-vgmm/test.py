import numpy as np
import matplotlib.pyplot as plt
import library.volumes.strmesh as vol
import scipy.io as sio
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn import mixture
import open3d as o3d
import os,fnmatch
import gdist

nolineal=1

tipo='holes'
xfilename='data/curvature/shrec16/'+tipo+'/'
gtfilename='data/gt/shrec16/'+tipo+'/'
components=2
montecarlo=200
gamma=1e-2

base='data/shrec2016/'

filesmodel=os.listdir(base+'null/')
pattern='*.mat'
for fimodel in filesmodel:
    if fnmatch.fnmatch(fimodel, pattern):
        model=fimodel.split('.')[0]
        print('----Este es el grupo:',model,'---')

        filesshape=os.listdir(base+tipo+'/')
        pattern2=tipo+'_'+model+'*.mat'
        for fishape in filesshape:
            if fnmatch.fnmatch(fishape, pattern2):
                shape=fishape.split('.')[0]
                data=shape+'.mat'
                print('----Forma:',shape,'----')