
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

from time import time
import tensorflow as tf
import numpy as np

import gpflowmodi
from ..utilities import ops, print_summary
from ..config import set_default_float, default_float

from .gplvm import BayesianGPLVM,VMGPLVM,BayesianVMGPLVM,MultiBayesianVMGPLVM


class RunGPLVM(BaseEstimator, TransformerMixin):
    def __init__(self,gpmethod='mvmgplvm', method='adam', learning_rate=1e-2, itera=3000, num_clusters=2, seed=1234, select_init='bgplvm', num_inducing=30, latent_dim=2, gpvariance=None, verbose=False):
        set_default_float(np.float64)
        self.gpmethod=gpmethod #mvmgplvm-vmgplvm-bvmgplvm
        self.method=method
        self.learning_rate=learning_rate
        self.itera=itera
        self.num_clusters=num_clusters
        self.seed=seed
        self.select_init=select_init #bgplvm-pca
        self.num_inducing=num_inducing
        self.latent_dim=latent_dim
        self.gpvariance=gpvariance
        self.verbose=verbose
        

    def run_keras(self,model, iterations, learning_rate, method):
        """
        Utility function running the keras optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        training_loss = model.training_loss_closure(compile=True)
        if method=="adam":
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate,amsgrad=True)
        elif method=="sgd":
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        elif method=="nadam":
            optimizer = tf.optimizers.Nadam(learning_rate=learning_rate)
        elif method=="adadelta":
            optimizer = tf.optimizers.Adadelta(learning_rate=learning_rate)
        elif method=="adagrad":
            optimizer = tf.optimizers.Adagrad(learning_rate=learning_rate)
        elif method=="adamax":
            optimizer = tf.optimizers.Adamax(learning_rate=learning_rate)
        elif method=="rmsprop":
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
                if self.verbose:
                    print('Iteration:',step)
        return logf

    def initialize(self, Y):
        X=ops.pca_reduce(Y, self.latent_dim)
        if self.select_init=='bgplvm':#BayesianGPLVM
            X_mean_init = X
            X_var_init = tf.random.uniform(shape=[self.num_data, self.latent_dim],dtype=default_float(),seed=self.seed)
            np.random.seed(self.seed)
            inducing_variable = tf.convert_to_tensor(
                np.random.permutation(X.numpy())[:self.num_inducing], dtype=default_float()
            )
            lengthscales = 10.0*tf.random.uniform(shape=[self.latent_dim],dtype=default_float(),seed=self.seed)
            kernel = gpflowmodi.kernels.RBF(lengthscales=lengthscales)

            gplvm = gpflowmodi.models.BayesianGPLVM(
                Y,
                X_data_mean=X_mean_init,
                X_data_var=X_var_init,
                kernel=kernel,
                inducing_variable=inducing_variable,
            )
            gplvm.likelihood.variance.assign(tf.random.uniform(shape=[],dtype=default_float(),seed=self.seed))
            logf=self.run_keras(gplvm,self.itera,self.learning_rate,method=self.method)
            X=tf.convert_to_tensor(StandardScaler().fit_transform(gplvm.X_data_mean),dtype=default_float())
        if self.gpmethod=='vmgplvm':
            gmm = mixture.BayesianGaussianMixture(n_components=self.num_clusters,covariance_type='diag').fit(X)
        else:
            gmm = mixture.BayesianGaussianMixture(n_components=self.num_clusters).fit(X)
        gamma=tf.transpose(tf.convert_to_tensor(gmm.predict_proba(X), dtype=default_float()))
        X_mean_init=X[None,:,:]*gamma[:,:,None]
        X_var_init=tf.random.uniform(shape=[self.num_clusters, self.num_data, self.latent_dim],dtype=default_float(),seed=self.seed)
        X_prior_mean=tf.convert_to_tensor(gmm.means_, dtype=default_float())
        X_prior_var=tf.convert_to_tensor(gmm.covariances_, dtype=default_float())
        pi_prior=tf.convert_to_tensor(gmm.weights_, dtype=default_float())

        return [gamma,X_mean_init,X_var_init,X_prior_mean,X_prior_var,pi_prior]

    def fit(self, Y):
        Y=StandardScaler().fit_transform(Y)
        self.Y = tf.convert_to_tensor(Y, dtype=default_float())
        self.num_data=Y.shape[0]
        [gamma,X_mean_init,X_var_init,X_prior_mean,X_prior_var,pi_prior]=self.initialize(Y)

        if self.gpmethod=='mvmgplvm':
            indices = tf.range(start=0, limit=self.num_data, dtype=tf.int32)
            self.shuffled_indices = tf.random.shuffle(indices)
            self.order_indices = tf.argsort(self.shuffled_indices)
            if self.num_data%2==0:
                multi_X_mean_init=tf.split(tf.gather(X_mean_init,self.shuffled_indices,axis=1), num_or_size_splits=2, axis=1)
                multi_X_var_init=tf.split(tf.gather(X_var_init,self.shuffled_indices,axis=1), num_or_size_splits=2, axis=1)
                multi_gamma=tf.split(tf.gather(gamma,self.shuffled_indices,axis=1), num_or_size_splits=2, axis=1)
                multi_Y=tf.split(tf.gather(Y,self.shuffled_indices,axis=0), num_or_size_splits=2, axis=0)
            else:
                mid=int((self.num_data-1)/2)
                multi_X_mean_init=tf.split(tf.gather(X_mean_init,self.shuffled_indices,axis=1), num_or_size_splits=[mid,mid+1], axis=1)
                multi_X_var_init=tf.split(tf.gather(X_var_init,self.shuffled_indices,axis=1), num_or_size_splits=[mid,mid+1], axis=1)
                multi_gamma=tf.split(tf.gather(gamma,self.shuffled_indices,axis=1), num_or_size_splits=[mid,mid+1], axis=1)
                multi_Y=tf.split(tf.gather(Y,self.shuffled_indices,axis=0), num_or_size_splits=[mid,mid+1], axis=0)

            np.random.seed(self.seed)
            multi_inducing_variable=[]
            multi_kernel=[]
            for i in range(2):
                lengthscales = 10.0*tf.random.uniform(shape=[self.latent_dim],dtype=default_float(),seed=self.seed)
                multi_kernel.append(gpflowmodi.kernels.RBF(lengthscales=lengthscales))
                multi_inducing_variable.append(tf.convert_to_tensor(
                    np.random.permutation(tf.math.reduce_sum(multi_X_mean_init[i]*multi_gamma[i][:,:,None],axis=0).numpy())[:self.num_inducing], dtype=default_float()
                ))

            self.model = gpflowmodi.models.MultiBayesianVMGPLVM(
                multi_Y,
                X_data_mean=multi_X_mean_init,
                X_data_var=multi_X_var_init,
                kernel=multi_kernel,
                gamma=multi_gamma,
                inducing_variable=multi_inducing_variable,
                X_prior_mean=X_prior_mean
            )
        else:
            np.random.seed(self.seed)
            inducing_variable = tf.convert_to_tensor(
                np.random.permutation(tf.math.reduce_sum(X_mean_init*gamma[:,:,None],axis=0).numpy())[:self.num_inducing], dtype=default_float()
            )

            lengthscales = 10.0*tf.random.uniform(shape=[self.latent_dim],dtype=default_float(),seed=self.seed)
            kernel = gpflowmodi.kernels.RBF(lengthscales=lengthscales)

            if self.gpmethod=='bvgplvm':
                self.model = gpflowmodi.models.BayesianVMGPLVM(
                    Y,
                    X_data_mean=X_mean_init,
                    X_data_var=X_var_init,
                    kernel=kernel,
                    gamma=gamma,
                    inducing_variable=inducing_variable,
                    X_prior_mean=X_prior_mean
                )
            else:
                self.model = gpflowmodi.models.VMGPLVM(
                    Y,
                    X_data_mean=X_mean_init,
                    X_data_var=X_var_init,
                    kernel=kernel,
                    gamma=gamma,
                    inducing_variable=inducing_variable,
                    X_prior_mean=X_prior_mean,
                    X_prior_var=X_prior_var,
                    pi_prior=pi_prior
                )

        if self.gpvariance==None:
            self.model.likelihood.variance.assign(tf.random.uniform(shape=[],dtype=default_float(),seed=self.seed))
        else:
            self.model.likelihood.variance.assign(self.gpvariance)

        start_time=time()
        self.logf=self.run_keras(self.model,self.itera,self.learning_rate,method=self.method)
        self.elapsed_time=time()-start_time

        return self