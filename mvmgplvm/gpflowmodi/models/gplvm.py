# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import tensorflow as tf

from .. import covariances, kernels, likelihoods
from ..base import Parameter
from ..config import default_float, default_jitter
from ..expectations import expectation
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..mean_functions import MeanFunction, Zero
from ..probability_distributions import DiagonalGaussian, DiagonalMixtureGaussian
from ..utilities import positive, to_default_float, triangular
from ..utilities.ops import pca_reduce
from ..utilities.ownbijectors import positiveNormalize,espectralTransform, positiveClip,BijNormalize,choleskyTransform, BijClip, TraceNormalize, McholInc
from .gpr import GPR
from .model import GPModel, MeanAndVariance
from .training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from .util import data_input_to_tensor, inducingpoint_wrapper
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    def __init__(
        self,
        data: OutputData,
        latent_dim: int,
        X_data_mean: Optional[tf.Tensor] = None,
        kernel: Optional[Kernel] = None,
        mean_function: Optional[MeanFunction] = None,
    ):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.

        :param data: y data matrix, size N (number of points) x D (dimensions)
        :param latent_dim: the number of latent dimensions (Q)
        :param X_data_mean: latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param mean_function: mean function, by default None.
        """
        if X_data_mean is None:
            X_data_mean = pca_reduce(data, latent_dim)

        num_latent_gps = X_data_mean.shape[1]
        if num_latent_gps != latent_dim:
            msg = "Passed in number of latent {0} does not match initial X {1}."
            raise ValueError(msg.format(latent_dim, num_latent_gps))

        if mean_function is None:
            mean_function = Zero()

        if kernel is None:
            kernel = kernels.SquaredExponential(lengthscales=tf.ones((latent_dim,)))

        if data.shape[1] < num_latent_gps:
            raise ValueError("More latent dimensions than observed.")

        gpr_data = (Parameter(X_data_mean), data_input_to_tensor(data))
        super().__init__(gpr_data, kernel, mean_function=mean_function)


class BayesianGPLVM(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        X_prior_mean=None,
        X_prior_var=None,
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        num_data, num_latent_gps = X_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        
        assert X_data_var.ndim == 2

        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.num_data = num_data
        self.output_dim = self.data.shape[-1]

        assert np.all(X_data_mean.shape == X_data_var.shape)
        assert X_data_mean.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert X_data_var.shape[0] == self.data.shape[0], "X var and Y must be same size."

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
            Z = tf.random.shuffle(X_data_mean)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent_gps

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        
        Y_data = self.data

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 =tf.reduce_sum(expectation(pX, self.kernel))
    
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
       
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_data_var = (
            self.X_data_var
            if self.X_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.X_data_var)
        )
        NQ = to_default_float(tf.size(self.X_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        return bound

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.

        Note: This model does not allow full output covariances.

        :param Xnew: points at which to predict
        """
        if full_output_cov:
            raise NotImplementedError

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), axis=0)
                - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError


class VMGPLVM(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        gamma= None,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        X_prior_mean=None,
        X_prior_var=None,
        pi_prior=None,
        train_prior=True,
        prior_covariance_type='diag'
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param gamma: responsabilities size K (number of clusters) x N
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        :param pi_prior: prior pi term, Q
        """
        self.prior_covariance_type=prior_covariance_type
        num_clusters, num_data, num_latent_gps = X_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        assert X_data_var.ndim == 3

        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        if pi_prior is None:
            pi_prior=0.5*tf.ones([num_clusters],dtype=default_float())

        if train_prior:
            self.pi=Parameter(pi_prior,transform=positiveNormalize())
        else:
            self.pi=pi_prior

        if gamma is None:
            gamma=0.5*tf.ones([num_clusters,num_data],dtype=default_float())
        gamma=tf.cast(gamma,dtype=default_float())
        self.gamma=Parameter(gamma,transform=positiveNormalize())

        self.num_data = num_data
        self.num_clusters=num_clusters
        self.output_dim = self.data.shape[-1]

        assert np.all(X_data_mean.shape == X_data_var.shape)
        assert X_data_mean.shape[1] == self.data.shape[0], "X mean and Y must be same size."
        assert X_data_var.shape[1] == self.data.shape[0], "X var and Y must be same size."

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
            Z = tf.random.shuffle(X_data_mean)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[2] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.random.uniform(shape=[self.num_clusters,self.num_latent_gps],dtype=default_float())
        if X_prior_var is None:
            if self.prior_covariance_type=='diag':
                X_prior_var = tf.random.uniform(shape=[self.num_clusters,self.num_latent_gps],dtype=default_float())
            elif self.prior_covariance_type=='spherical':
                X_prior_var = tf.random.uniform(shape=[self.num_clusters],dtype=default_float())
            elif self.prior_covariance_type=='full':
                X_prior_var = tf.eye(self.num_latent_gps,batch_shape=[self.num_clusters],dtype=default_float())

        if train_prior:
            self.X_prior_mean = Parameter(tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float()))
            if self.prior_covariance_type=='diag' or self.prior_covariance_type=='spherical':
                self.X_prior_var = Parameter(tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float()), transform=positiveClip())
            elif self.prior_covariance_type=='full':
                X_prior_var=choleskyTransform().forward(X_prior_var+0.1)
                self.X_prior_var = Parameter(tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float()))
        else:
            self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
            self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        self.it=0
        assert self.X_prior_mean.shape[0] == self.num_clusters
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        # assert self.X_prior_var.shape[0] == self.num_clusters
        # assert self.X_prior_var.shape[1] == self.num_latent_gps
        # if self.prior_covariance_type=='full':
        #     assert self.X_prior_var.shape[2] == self.num_latent_gps

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        
        Y_data = self.data

        pX = DiagonalMixtureGaussian(self.X_data_mean, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 =tf.reduce_sum(self.gamma*expectation(pX, self.kernel))
        
        psi1 = tf.math.reduce_sum(tf.expand_dims(self.gamma, 2)*expectation(pX, (self.kernel, self.inducing_variable)),axis=0)
        
        psi2 =tf.reduce_sum( 
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.gamma, 2),3)*expectation(
                        pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
                    ),
                    axis=0
                ),
                axis=0
            )

        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        N=to_default_float(tf.shape(Y_data)[0])
        Q = to_default_float(tf.shape(self.X_data_mean)[2])
        K = to_default_float(tf.shape(self.X_data_mean)[0])
        if self.prior_covariance_type=='diag':
            Rchol=tf.tile(tf.expand_dims(tf.linalg.cholesky(tf.linalg.diag(self.X_prior_var)),1),[1,N,1,1])
        elif self.prior_covariance_type=='spherical':
            Rchol=tf.tile(tf.expand_dims(tf.linalg.cholesky(self.X_prior_var[:,None,None]*tf.eye(Q,batch_shape=[K],dtype=default_float())),1),[1,N,1,1])
        elif self.prior_covariance_type=='full':
            prior_var=tfp.math.fill_triangular(self.X_prior_var)
            Rchol=tf.tile(tf.expand_dims(prior_var,1),[1,N,1,1])
        prior_mean=tf.expand_dims(self.X_prior_mean,1)

        D = to_default_float(tf.shape(Y_data)[1])

        # # KL[q(x|z) || p(x|z)]_q(z)
        d=tf.expand_dims(self.X_data_mean-prior_mean,3)
        Aklchol=tf.linalg.triangular_solve(Rchol,d, lower=True)#tf.math.reduce_sum(tf.math.multiply(tf.einsum('knij,knj->kni',Rinv,d),d),axis=-1)
        Akl=tf.math.reduce_sum(tf.math.reduce_sum(Aklchol*Aklchol,axis=-1),axis=-1)

        Lambdadi=tf.linalg.diag(self.X_data_var)
        Bkltmp=tf.linalg.triangular_solve(Rchol,Lambdadi, lower=True)
        Bklchol=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(Rchol),Bkltmp, lower=False)
        Bkl=tf.linalg.trace(Bklchol)

        Rcholdiag=tf.linalg.diag_part(Rchol)
        F=tf.math.log(tf.math.reduce_prod(Rcholdiag*Rcholdiag,axis=-1))-tf.reduce_sum(tf.math.log(self.X_data_var),axis=-1)
        
        Dkl=F-Q+Bkl+Akl
        KLx=0.5*tf.math.reduce_sum(self.gamma*Dkl)

        # KL[q(z) || p(z)]
        KLz=tf.math.reduce_sum(self.gamma*(tf.math.log(self.gamma)-tf.math.log(tf.expand_dims(self.pi,1))))

        # D = to_default_float(tf.shape(Y_data)[1])
        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KLx
        bound -= KLz

        return bound

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

class BayesianVMGPLVM(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        gamma= None,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        X_prior_mean=None
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param gamma: responsabilities size K (number of clusters) x N
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        :param pi_prior: prior pi term, Q
        """
        num_clusters, num_data, num_latent_gps = X_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        assert X_data_var.ndim == 3

        #priors params
        self.eta0=tf.ones(num_clusters,dtype=default_float())#*1/num_clusters
        self.w0=tf.math.reduce_mean(tf.math.reduce_sum(X_data_mean*gamma[:,:,None],axis=0), axis=0)#tf.zeros(num_latent_gps,dtype=default_float())tf.math.reduce_mean(tf.math.reduce_sum(X_data_mean*gamma[:,:,None],axis=0), axis=0)
        self.S0=tf.eye(num_latent_gps,dtype=default_float())
        self.zeta0=to_default_float(num_latent_gps)
        self.theta0=to_default_float(1.0)

        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        if gamma is None:
            gamma=0.5*tf.ones([num_clusters,num_data],dtype=default_float())
        gamma=tf.cast(gamma,dtype=default_float())
        self.gamma=Parameter(gamma,transform=positiveNormalize())

        Nk=tf.linalg.normalize((num_data/num_clusters)+tf.zeros(num_clusters,dtype=default_float()),ord=1)[0]#
        self.eta =Parameter(self.eta0+Nk, transform=positive())
        self.theta=Parameter(self.theta0+Nk, transform=positive())
        self.w=Parameter(X_prior_mean)
        Sinit=choleskyTransform().forward(self.S0+0.1)
        self.S=Parameter(tf.tile(tf.expand_dims(Sinit,0), (num_clusters,1)))#Parameter(tf.tile(tf.expand_dims(self.S0,0), (num_clusters,1,1)),transform=McholInc())
        self.zeta=Parameter(self.zeta0+Nk, transform=positive(lower=num_latent_gps))

        self.num_data = num_data
        self.num_clusters=num_clusters
        self.output_dim = self.data.shape[-1]

        assert np.all(X_data_mean.shape == X_data_var.shape)
        assert X_data_mean.shape[1] == self.data.shape[0], "X mean and Y must be same size."
        assert X_data_var.shape[1] == self.data.shape[0], "X var and Y must be same size."

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
            Z = tf.random.shuffle(X_data_mean)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[2] == self.num_latent_gps

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        
        Y_data = self.data

        pX = DiagonalMixtureGaussian(self.X_data_mean, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 =tf.reduce_sum(self.gamma*expectation(pX, self.kernel))
        
        psi1 = tf.math.reduce_sum(tf.expand_dims(self.gamma, 2)*expectation(pX, (self.kernel, self.inducing_variable)),axis=0)
        
        psi2 =tf.reduce_sum( 
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.gamma, 2),3)*expectation(
                        pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
                    ),
                    axis=0
                ),
                axis=0
            )

        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        D = to_default_float(tf.shape(Y_data)[1])
        N=to_default_float(tf.shape(Y_data)[0])
        Q = to_default_float(tf.shape(self.X_data_mean)[2])
        #K = to_default_float(tf.shape(self.X_data_mean)[0])

        
        #KL1
        schol=tfp.math.fill_triangular(self.S)#tf.linalg.cholesky(self.S)
        scholtile=tf.tile(tf.expand_dims(schol,axis=1),[1,N,1,1])
        scholinv=tf.linalg.inv(schol)
        scholinvtile=tf.tile(tf.expand_dims(scholinv,axis=1),[1,N,1,1])

        Kl11=tf.math.reduce_sum(tf.math.digamma(0.5*(self.zeta[:,None]-tf.range(0,Q,dtype=default_float())[None,:])),axis=-1)
        Kl12=2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(schol)),axis=-1)#tf.linalg.slogdet(self.S)[1]#tf.math.log(tf.linalg.det(self.S))
        
        Kl13=tf.reduce_sum(tf.math.log(self.X_data_var),axis=-1)
        Kl14=((1.0/self.theta)-to_default_float(tf.math.log(2.0))-to_default_float(1.0))*Q

        Lambdadi=tf.linalg.diag(self.X_data_var)
        Kl15tmp=tf.linalg.matmul(tf.linalg.matrix_transpose(scholtile),Lambdadi)
        Kl15=tf.linalg.trace(self.zeta[:,None,None,None]*tf.linalg.triangular_solve(scholinvtile,Kl15tmp,lower=True))

        d=tf.expand_dims(self.X_data_mean-self.w[:,None,:],axis=3)
        Kl16tmp=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(scholinvtile),d, lower=False)
        Kl16=self.zeta[:,None]*tf.math.reduce_sum(tf.math.reduce_sum(Kl16tmp*Kl16tmp,axis=-1),axis=-1)#self.zeta[:,None]*tf.math.reduce_sum(tf.math.multiply(tf.einsum('kij,knj->kni',self.S,d),d),axis=-1)
        Kl1=-Kl11[:,None]-Kl12[:,None]-Kl13+Kl14[:,None]+Kl15+Kl16
        Kl1*=self.gamma
        Kl1=0.5*tf.math.reduce_sum(Kl1)
        
        

        #KL2
        dihatet=tf.math.digamma(tf.math.reduce_sum(self.eta))
        Kl2=tf.math.reduce_sum(self.gamma*(tf.math.log(self.gamma)+dihatet-tf.math.digamma(self.eta)[:,None]))
        
        #KL3
        lgamahatet=tf.math.lgamma(tf.math.reduce_sum(self.eta))
        lgamahatet0=tf.math.lgamma(tf.math.reduce_sum(self.eta0))
        Kl31=lgamahatet-lgamahatet0
        Kl32=tf.math.lgamma(self.eta0)-tf.math.lgamma(self.eta)
        Kl33=(self.eta-self.eta0)*(tf.math.digamma(self.eta)-dihatet)
        Kl3=Kl31+tf.math.reduce_sum(Kl32+Kl33)

        #KL4
        Kl4d=tf.expand_dims(self.w0[None,:]-self.w,axis=2)
        Kl14tmp=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(scholinv),Kl4d, lower=False)
        Kl41=self.theta0*self.zeta*tf.math.reduce_sum(tf.math.reduce_sum(Kl14tmp*Kl14tmp,axis=-1),axis=-1)#tf.math.reduce_sum(tf.math.multiply(tf.einsum('kij,kj->ki',self.S,Kl4d),Kl4d),axis=-1)
        Kl42=((self.theta0/self.theta)+tf.math.log(self.theta)-tf.math.log(self.theta0)-to_default_float(1.0))*Q
        Kl4=0.5*tf.math.reduce_sum(Kl41+Kl42)
        

        #KL5
        Kl51=self.zeta0*(tf.linalg.slogdet(self.S0)[1]-2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(schol)),axis=-1))
        Kl52tmp=tf.linalg.matmul(tf.linalg.matrix_transpose(schol),tf.linalg.inv(self.S0))
        Kl52=self.zeta*tf.linalg.trace(tf.linalg.triangular_solve(scholinv,Kl52tmp,lower=True))#tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(self.S0),self.S))
        Kl53=2.0*(tf.math.lgamma(self.zeta0/2.0)-tf.math.lgamma(self.zeta/2.0))
        Kl54=(self.zeta-self.zeta0)*tf.math.digamma(self.zeta/2.0)-self.zeta*Q
        Kl5=0.5*tf.math.reduce_sum(Kl51+Kl52+Kl53+Kl54)

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= Kl1
        bound -= Kl2
        bound -= Kl3
        bound -= Kl4
        bound -= Kl5

        return bound

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

class MultiBayesianVMGPLVM(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data_mean: None,
        X_data_var: None,
        kernel: None,
        gamma= None,
        inducing_variable=None,
        X_prior_mean=None
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param gamma: responsabilities size K (number of clusters) x N
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        :param pi_prior: prior pi term, Q
        """
        num_views=len(X_data_mean)
        num_clusters, num_data1, num_latent_gps = X_data_mean[0].shape
        num_clusters, num_data2, num_latent_gps = X_data_mean[0].shape
        num_data=num_data1+num_data2
        self.kernel0 = kernel[0]
        self.kernel1 = kernel[1]
        self.likelihood = likelihoods.Gaussian()
        
        self.data0 = data_input_to_tensor(data[0])
        self.data1 = data_input_to_tensor(data[1])

        #priors params
        self.eta0=tf.ones(num_clusters,dtype=default_float())#*1/num_clusters
        self.w0=tf.zeros(num_latent_gps,dtype=default_float())#tf.math.reduce_mean(tf.math.reduce_sum(X_data_mean*gamma[:,:,None],axis=0), axis=0)
        self.S0=tf.eye(num_latent_gps,dtype=default_float())
        self.zeta0=to_default_float(num_latent_gps)
        self.theta0=to_default_float(1.0)

        self.X_data_mean0 = Parameter(X_data_mean[0])
        self.X_data_var0 = Parameter(X_data_var[0], transform=positive())
        self.X_data_mean1 = Parameter(X_data_mean[1])
        self.X_data_var1 = Parameter(X_data_var[1], transform=positive())

        self.gamma0=Parameter(gamma[0],transform=positiveNormalize())
        self.gamma1=Parameter(gamma[1],transform=positiveNormalize())

        Nk=tf.linalg.normalize((num_data/num_clusters)+tf.zeros(num_clusters,dtype=default_float()),ord=1)[0]#
        self.eta =Parameter(self.eta0+Nk, transform=positive())
        self.theta=Parameter(self.theta0+Nk, transform=positive())
        self.w=Parameter(X_prior_mean)
        Sinit=choleskyTransform().forward(self.S0+0.1)
        self.S=Parameter(tf.tile(tf.expand_dims(Sinit,0), (num_clusters,1)))#Parameter(tf.tile(tf.expand_dims(self.S0,0), (num_clusters,1,1)),transform=McholInc())
        self.zeta=Parameter(self.zeta0+Nk, transform=positive(lower=num_latent_gps))

        self.inducing_variable0 = inducingpoint_wrapper(inducing_variable[0])
        self.inducing_variable1 = inducingpoint_wrapper(inducing_variable[1])


    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        
        Y_data = self.data0

        pX = DiagonalMixtureGaussian(self.X_data_mean0, self.X_data_var0)

        num_inducing = self.inducing_variable0.num_inducing
        psi0 =tf.reduce_sum(self.gamma0*expectation(pX, self.kernel0))
        
        psi1 = tf.math.reduce_sum(tf.expand_dims(self.gamma0, 2)*expectation(pX, (self.kernel0, self.inducing_variable0)),axis=0)
        
        psi2 =tf.reduce_sum( 
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.gamma0, 2),3)*expectation(
                        pX, (self.kernel0, self.inducing_variable0), (self.kernel0, self.inducing_variable0)
                    ),
                    axis=0
                ),
                axis=0
            )

        cov_uu = covariances.Kuu(self.inducing_variable0, self.kernel0, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        D = to_default_float(tf.shape(Y_data)[1])
        N=to_default_float(tf.shape(Y_data)[0])
        Q = to_default_float(tf.shape(self.X_data_mean0)[2])
        #K = to_default_float(tf.shape(self.X_data_mean0)[0])

        
        #KL1
        schol=tfp.math.fill_triangular(self.S)#tf.linalg.cholesky(self.S)
        scholtile=tf.tile(tf.expand_dims(schol,axis=1),[1,N,1,1])
        scholinv=tf.linalg.inv(schol)
        scholinvtile=tf.tile(tf.expand_dims(scholinv,axis=1),[1,N,1,1])

        Kl11=tf.math.reduce_sum(tf.math.digamma(0.5*(self.zeta[:,None]-tf.range(0,Q,dtype=default_float())[None,:])),axis=-1)
        Kl12=2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(schol)),axis=-1)#tf.linalg.slogdet(self.S)[1]#tf.math.log(tf.linalg.det(self.S))
        
        Kl13=tf.reduce_sum(tf.math.log(self.X_data_var0),axis=-1)
        Kl14=((1.0/self.theta)-to_default_float(tf.math.log(2.0))-to_default_float(1.0))*Q

        Lambdadi=tf.linalg.diag(self.X_data_var0)
        Kl15tmp=tf.linalg.matmul(tf.linalg.matrix_transpose(scholtile),Lambdadi)
        Kl15=tf.linalg.trace(self.zeta[:,None,None,None]*tf.linalg.triangular_solve(scholinvtile,Kl15tmp,lower=True))

        d=tf.expand_dims(self.X_data_mean0-self.w[:,None,:],axis=3)
        Kl16tmp=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(scholinvtile),d, lower=False)
        Kl16=self.zeta[:,None]*tf.math.reduce_sum(tf.math.reduce_sum(Kl16tmp*Kl16tmp,axis=-1),axis=-1)#self.zeta[:,None]*tf.math.reduce_sum(tf.math.multiply(tf.einsum('kij,knj->kni',self.S,d),d),axis=-1)
        Kl1=-Kl11[:,None]-Kl12[:,None]-Kl13+Kl14[:,None]+Kl15+Kl16
        Kl1*=self.gamma0
        Kl1=0.5*tf.math.reduce_sum(Kl1)
        
        

        #KL2
        dihatet=tf.math.digamma(tf.math.reduce_sum(self.eta))
        Kl2=tf.math.reduce_sum(self.gamma0*(tf.math.log(self.gamma0)+dihatet-tf.math.digamma(self.eta)[:,None]))
        

        
        

       

        # compute log marginal bound0
        ND = to_default_float(tf.size(Y_data))
        bound0 = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound0 += -0.5 * D * log_det_B
        bound0 += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound0 += 0.5 * tf.reduce_sum(tf.square(c))
        bound0 += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound0 -= Kl1
        bound0 -= Kl2


        Y_data = self.data1

        pX = DiagonalMixtureGaussian(self.X_data_mean1, self.X_data_var1)

        num_inducing = self.inducing_variable1.num_inducing
        psi0 =tf.reduce_sum(self.gamma1*expectation(pX, self.kernel1))
        
        psi1 = tf.math.reduce_sum(tf.expand_dims(self.gamma1, 2)*expectation(pX, (self.kernel1, self.inducing_variable1)),axis=0)
        
        psi2 =tf.reduce_sum( 
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.gamma1, 2),3)*expectation(
                        pX, (self.kernel1, self.inducing_variable1), (self.kernel1, self.inducing_variable1)
                    ),
                    axis=0
                ),
                axis=0
            )

        cov_uu = covariances.Kuu(self.inducing_variable1, self.kernel1, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        D = to_default_float(tf.shape(Y_data)[1])
        N=to_default_float(tf.shape(Y_data)[0])
        Q = to_default_float(tf.shape(self.X_data_mean1)[2])
        #K = to_default_float(tf.shape(self.X_data_mean1)[0])

        
        #KL1
        schol=tfp.math.fill_triangular(self.S)#tf.linalg.cholesky(self.S)
        scholtile=tf.tile(tf.expand_dims(schol,axis=1),[1,N,1,1])
        scholinv=tf.linalg.inv(schol)
        scholinvtile=tf.tile(tf.expand_dims(scholinv,axis=1),[1,N,1,1])

        Kl11=tf.math.reduce_sum(tf.math.digamma(0.5*(self.zeta[:,None]-tf.range(0,Q,dtype=default_float())[None,:])),axis=-1)
        Kl12=2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(schol)),axis=-1)#tf.linalg.slogdet(self.S)[1]#tf.math.log(tf.linalg.det(self.S))
        
        Kl13=tf.reduce_sum(tf.math.log(self.X_data_var1),axis=-1)
        Kl14=((1.0/self.theta)-to_default_float(tf.math.log(2.0))-to_default_float(1.0))*Q

        Lambdadi=tf.linalg.diag(self.X_data_var1)
        Kl15tmp=tf.linalg.matmul(tf.linalg.matrix_transpose(scholtile),Lambdadi)
        Kl15=tf.linalg.trace(self.zeta[:,None,None,None]*tf.linalg.triangular_solve(scholinvtile,Kl15tmp,lower=True))

        d=tf.expand_dims(self.X_data_mean1-self.w[:,None,:],axis=3)
        Kl16tmp=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(scholinvtile),d, lower=False)
        Kl16=self.zeta[:,None]*tf.math.reduce_sum(tf.math.reduce_sum(Kl16tmp*Kl16tmp,axis=-1),axis=-1)#self.zeta[:,None]*tf.math.reduce_sum(tf.math.multiply(tf.einsum('kij,knj->kni',self.S,d),d),axis=-1)
        Kl1=-Kl11[:,None]-Kl12[:,None]-Kl13+Kl14[:,None]+Kl15+Kl16
        Kl1*=self.gamma1
        Kl1=0.5*tf.math.reduce_sum(Kl1)

        #KL2
        dihatet=tf.math.digamma(tf.math.reduce_sum(self.eta))
        Kl2=tf.math.reduce_sum(self.gamma1*(tf.math.log(self.gamma1)+dihatet-tf.math.digamma(self.eta)[:,None]))
        
        #KL3
        lgamahatet=tf.math.lgamma(tf.math.reduce_sum(self.eta))
        lgamahatet0=tf.math.lgamma(tf.math.reduce_sum(self.eta0))
        Kl31=lgamahatet-lgamahatet0
        Kl32=tf.math.lgamma(self.eta0)-tf.math.lgamma(self.eta)
        Kl33=(self.eta-self.eta0)*(tf.math.digamma(self.eta)-dihatet)
        Kl3=Kl31+tf.math.reduce_sum(Kl32+Kl33)

        #KL4
        Kl4d=tf.expand_dims(self.w0[None,:]-self.w,axis=2)
        Kl14tmp=tf.linalg.triangular_solve(tf.linalg.matrix_transpose(scholinv),Kl4d, lower=False)
        Kl41=self.theta0*self.zeta*tf.math.reduce_sum(tf.math.reduce_sum(Kl14tmp*Kl14tmp,axis=-1),axis=-1)#tf.math.reduce_sum(tf.math.multiply(tf.einsum('kij,kj->ki',self.S,Kl4d),Kl4d),axis=-1)
        Kl42=((self.theta0/self.theta)+tf.math.log(self.theta)-tf.math.log(self.theta0)-to_default_float(1.0))*Q
        Kl4=0.5*tf.math.reduce_sum(Kl41+Kl42)
        

        #KL5
        Kl51=self.zeta0*(tf.linalg.slogdet(self.S0)[1]-2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(schol)),axis=-1))
        Kl52tmp=tf.linalg.matmul(tf.linalg.matrix_transpose(schol),tf.linalg.inv(self.S0))
        Kl52=self.zeta*tf.linalg.trace(tf.linalg.triangular_solve(scholinv,Kl52tmp,lower=True))#tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(self.S0),self.S))
        Kl53=2.0*(tf.math.lgamma(self.zeta0/2.0)-tf.math.lgamma(self.zeta/2.0))
        Kl54=(self.zeta-self.zeta0)*tf.math.digamma(self.zeta/2.0)-self.zeta*Q
        Kl5=0.5*tf.math.reduce_sum(Kl51+Kl52+Kl53+Kl54)

        # compute log marginal bound1
        ND = to_default_float(tf.size(Y_data))
        bound1 = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound1 += -0.5 * D * log_det_B
        bound1 += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound1 += 0.5 * tf.reduce_sum(tf.square(c))
        bound1 += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound1 -= Kl1
        bound1 -= Kl2

        totalBound= bound0+bound1
        totalBound -= Kl3
        totalBound -= Kl4
        totalBound -= Kl5

        return totalBound

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError