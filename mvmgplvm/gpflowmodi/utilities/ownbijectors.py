import tensorflow_probability as tfp
import tensorflow as tf
from ..config import default_float
from typing import Optional
from .bijectors import positive
from .misc import to_default_float

class BijNormalize(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='bijnormalize', axis=0):
        super(BijNormalize,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name,
            dtype=default_float())
        self.axis=axis

    def _forward(self, x):
        return tf.linalg.normalize(x,ord=1,axis=self.axis)[0]

    def _inverse(self, y):
        return y

    # def _inverse_log_det_jacobian(self, y):
    #   return -self._forward_log_det_jacobian(self._inverse(y))

    # def _forward_log_det_jacobian(self, x):
    #   # Notice that we needn't do any reducing, even when`event_ndims > 0`.
    #   # The base Bijector class will handle reducing for us; it knows how
    #   # to do so because we called `super` `__init__` with
    #   # `forward_min_event_ndims = 0`.
    #   return x

class BijTotalNormalize(tfp.bijectors.Bijector):

    def __init__(self, validate_args=False, name='bijtotalnormalize'):
        super(BijTotalNormalize,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name,
            dtype=default_float())

    def _forward(self, x):
        return tf.linalg.normalize(x,ord=1)[0]

    def _inverse(self, y):
        return y

    # def _inverse_log_det_jacobian(self, y):
    #   return -self._forward_log_det_jacobian(self._inverse(y))

    # def _forward_log_det_jacobian(self, x):
    #   # Notice that we needn't do any reducing, even when`event_ndims > 0`.
    #   # The base Bijector class will handle reducing for us; it knows how
    #   # to do so because we called `super` `__init__` with
    #   # `forward_min_event_ndims = 0`.
    #   return x

class EspectralBij(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='EspectralBij', clip_value_max=1000, tol=1e-12):
        super(EspectralBij,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
        self.clip_value_max=clip_value_max
        self.tol=tol
    def _forward(self, x):
        # x=x/tf.linalg.trace(x)[None,:]
        eigen_values_Q, eigen_vectors_Q = tf.linalg.eigh(x)
        new_diag_lambda = tf.linalg.diag(tf.clip_by_value(eigen_values_Q,clip_value_min=self.tol,clip_value_max=self.clip_value_max))
        new_x=tf.linalg.matmul(tf.linalg.matmul(eigen_vectors_Q, new_diag_lambda), tf.linalg.matrix_transpose(eigen_vectors_Q))
        
        # Q = to_default_float(tf.shape(x)[2])
        # I=tf.expand_dims(tf.eye(Q,dtype=default_float()),axis=0)
        
        # tole=tf.constant(self.tol,dtype=default_float()) #tol
        # new_x=tf.add(new_x,tole*I)

        

        return new_x

    def _inverse(self, y):
        return y


class McholInc(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='McholInc', tol=1e-12):
        super(McholInc,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
        self.tol=tol
    
    @tf.function
    def _forward(self, S):
        K=S.shape[0]
        Q=S.shape[1]
        beta=tf.norm(S,ord='fro',axis=[1,2])
        tol=tf.constant(1e-12,dtype=default_float())
        mayor=tf.math.reduce_min(tf.linalg.diag_part(S),axis=-1)>tol
        tau=tf.zeros([K],dtype=default_float())
        tau=tf.where(mayor,tau,tf.math.maximum(beta/2,tol))
        vali=tf.constant(1,dtype=default_float())
        while(tf.equal(vali,vali)):
            I=tf.expand_dims(tf.eye(Q,dtype=default_float()),axis=0)
            new_S=tf.add(S,tau[:,None,None]*I)
            schol=tf.linalg.cholesky(new_S)
            whonan=~tf.math.is_nan(tf.math.reduce_min(tf.math.reduce_min(schol,axis=-1),axis=-1))
            val=tf.math.reduce_all(whonan)
            if tf.equal(val,True):
                break
            tau=tf.where(whonan,tau,tf.math.maximum(2*tau,beta/2))

        return new_S

    def _inverse(self, y):
        return y

class OrthBij(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='OrthBij'):
        super(OrthBij,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
      
    def _forward(self, x):
        return tfp.math.gram_schmidt(x)

    def _inverse(self, y):
        return y

class SymmBij(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='SymmBij'):
        super(SymmBij,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
      
    def _forward(self, x):
        X_upper = tf.linalg.band_part(x, 0, -1)
        sym_matrix_A = tf.multiply(tf.constant(0.5,dtype=default_float()), (X_upper + tf.linalg.matrix_transpose(X_upper)))
        return sym_matrix_A

    def _inverse(self, y):
        return y

class SymmBij(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='SymmBij'):
        super(SymmBij,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
      
    def _forward(self, x):
        X_upper = tf.linalg.band_part(x, 0, -1)
        sym_matrix_A = tf.multiply(tf.constant(0.5,dtype=default_float()), (X_upper + tf.linalg.matrix_transpose(X_upper)))
        return sym_matrix_A

    def _inverse(self, y):
        return y

class BijClip(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='BijClip',mini=-1,maxi=1):
        super(BijClip,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
        self.mini=mini
        self.maxi=maxi=maxi
      
    def _forward(self, x):
        return tf.clip_by_value(x,clip_value_min=self.mini,clip_value_max=self.maxi)

    def _inverse(self, y):
        return y

class TraceNormalize(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='TraceNormalize'):
        super(TraceNormalize,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            dtype=default_float(),
            name=name)
      
    def _forward(self, x):
        x_new=tf.linalg.normalize(x,ord=1,axis=1)[0]
        return x_new

    def _inverse(self, y):
        return y


def positiveNormalize(lower: Optional[float] = None, base: Optional[str] = None) -> tfp.bijectors.Bijector:
    bijector=BijNormalize()
    return tfp.bijectors.Chain([bijector,positive()])

def positiveClip(lower: Optional[float] = None, base: Optional[str] = None) -> tfp.bijectors.Bijector:
    bijector=BijClip()
    return tfp.bijectors.Chain([positive()])

def espectralTransform(lower: Optional[float] = None, base: Optional[str] = None) -> tfp.bijectors.Bijector:
    return tfp.bijectors.Chain([EspectralBij()])

def choleskyTransform() -> tfp.bijectors.Bijector:
    VALIDATE_ARGS=True
    return tfp.bijectors.Chain([
        # step 3: flatten the lower triangular portion of the matrix
        tfp.bijectors.Invert(tfp.bijectors.FillTriangular(validate_args=VALIDATE_ARGS)),
        # step 2: take the log of the diagonals    
        # tfp.bijectors.CorrelationCholesky(),
        tfp.bijectors.TransformDiagonal(tfp.bijectors.Invert(tfp.bijectors.Exp(validate_args=VALIDATE_ARGS))),
        # step 1: decompose the precision matrix into its Cholesky factors
        tfp.bijectors.Invert(tfp.bijectors.CholeskyOuterProduct(validate_args=VALIDATE_ARGS)),
    ])

def logDiagTransform() -> tfp.bijectors.Bijector:
    VALIDATE_ARGS=True
    return tfp.bijectors.Chain([
        # step 3: flatten the lower triangular portion of the matrix
        tfp.bijectors.Invert(tfp.bijectors.FillTriangular(validate_args=VALIDATE_ARGS)),
        # step 2: take the log of the diagonals    
        # tfp.bijectors.CorrelationCholesky(),
        tfp.bijectors.TransformDiagonal(tfp.bijectors.Invert(tfp.bijectors.Exp(validate_args=VALIDATE_ARGS))),
        # step 1: decompose the precision matrix into its Cholesky factors
        tfp.bijectors.FillTriangular(validate_args=VALIDATE_ARGS),
    ])

def fillTriangular() -> tfp.bijectors.Bijector:
    VALIDATE_ARGS=True
    return tfp.bijectors.Invert(tfp.bijectors.FillTriangular(validate_args=VALIDATE_ARGS))
