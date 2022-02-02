import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn import mixture
import os,fnmatch
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
nolineal=1
xfilename='data/curvature/tosca/'
components=4
montecarlo=200
gamma=0.01
base='data/tosca/'

files=os.listdir(xfilename)
pattern='*.mat'
for fi in files:
    if fnmatch.fnmatch(fi, pattern):
        print('----Este es el archivo:',fi,'---')
        namefi=fi.split('.')[0]
        try:
            filename=xfilename+namefi+'.mat'
            info=sio.whosmat(filename)[0]
            mat=sio.loadmat(filename)
            X=mat[info[0]]

            if nolineal==1:
                #gmm-no lineal
                steps = [('rff', RBFSampler(gamma=gamma,n_components=montecarlo)), 
                        ('cluster', mixture.BayesianGaussianMixture(n_components=components))] #clasificador 
                method = Pipeline(steps)
            else:
                #gmm lineal
                steps = [ ('cluster', mixture.BayesianGaussianMixture(n_components=components, random_state=48))] #clasificador 
                method = Pipeline(steps)

            print('----Inicio Entrenamiento-----')
            method.fit(X)
            Z=method.predict(X)
            print('----Fin Entrenamiento----')

            tam1=int(X.shape[0]/2)
            print(tam1)
            Z1=Z[0:tam1]
            Z2=Z[tam1:]

            ars=adjusted_rand_score(Z1,Z2)
            rs=rand_score(Z1,Z2)
            metrics=np.array([[ars,rs]])
            print('---Finaliza random scores---')
            saveruta='resultsgmm-nolineal/randscore-tosca/'+namefi+'.txt'

            np.savetxt(saveruta, metrics, fmt='%f')
        except:
            print('Ha surgido un error en '+surface1+'-'+surface2)