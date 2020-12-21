from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

# default initializer
def default_initializer(std=0.05):
  return tf.random_normal_initializer(0., std)

# This masks part of a matrix to stop backpropagation.
def entry_stop_gradients(target, mask):
    """
    mask specify which entries to be trained
    """
    mask_stop = tf.logical_not(mask)
    mask = tf.cast(mask, dtype=target.dtype)
    mask_stop = tf.cast(mask_stop, dtype=target.dtype)

    return tf.stop_gradient(mask_stop * target) + mask * target

# This flow_mapping is for infinite support
# stacking actnorm, and affince coupling layers.
class flow_mapping(layers.Layer):
  def __init__(self, name, n_depth, n_split_at, n_width = 32, flow_coupling = 0, **kwargs):
    super(flow_mapping, self).__init__(name=name,**kwargs)
    self.n_depth = n_depth
    self.n_split_at = n_split_at
    self.n_width = n_width
    self.flow_coupling = flow_coupling

    # two affine coupling layers are needed for each update of the whole vector
    assert n_depth % 2 == 0

  def build(self, input_shape):
    self.n_length = input_shape[-1]
    self.scale_layers = []
    self.affine_layers = []

    sign = -1
    for i in range(self.n_depth):
      self.scale_layers.append(actnorm('actnorm_' + str(i)))
      sign *= -1
      i_split_at = (self.n_split_at*sign + self.n_length) % self.n_length
      self.affine_layers.append(affine_coupling('af_coupling_' + str(i),
                                                i_split_at,
                                                n_width=self.n_width,
                                                flow_coupling=self.flow_coupling))

  # the default setting mapping the given data to the prior distribution
  # without computing the jacobian.
  def call(self, inputs, logdet=None, reverse=False):
    if not reverse:
      z = inputs
      for i in range(self.n_depth):
        z = self.scale_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = self.affine_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = z[:,::-1]
    else:
      z = inputs
      for i in reversed(range(self.n_depth)):
        z = z[:,::-1]

        z = self.affine_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

        z = self.scale_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

    if logdet is not None:
        return z, logdet
    return z

  def actnorm_data_initialization(self):
    for i in range(self.n_depth):
        self.scale_layers[i].reset_data_initialization()

# optimal linear mapping for least important dimension - rotation layer
class W_LU(layers.Layer):
  def __init__(self, name, **kwargs):
    super(W_LU, self).__init__(name=name, **kwargs)

  def build(self, input_shape):
    self.n_length = input_shape[-1]

    # lower- and upper-triangluar parts in one matrix.
    self.LU = self.add_weight(name='LU', shape=(self.n_length, self.n_length),
                              initializer=tf.zeros_initializer(),
                              dtype=tf.float32, trainable=True)

    # identity matrix
    self.LU_init = tf.eye(self.n_length, dtype=tf.float32)

  def call(self, inputs, logdet = None, reverse=False):
    x = inputs
    n_dim = x.shape[-1]

    # L*U*x
    LU = self.LU_init + self.LU

    # upper-triangular matrix
    U = tf.linalg.band_part(LU,0,-1)

    # diagonal line
    U_diag = tf.linalg.tensor_diag_part(U)

    # trainable mask for U
    U_mask = (tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0,-1) >= 1)
    U = entry_stop_gradients(U, U_mask)

    # lower-triangular matrix
    I = tf.eye(self.n_length,dtype=tf.float32)
    L = tf.linalg.band_part(I+LU,-1,0)-tf.linalg.band_part(LU,0,0)

    # trainable mask for L
    L_mask = (tf.linalg.band_part(tf.ones([n_dim, n_dim]), -1, 0) - tf.linalg.band_part(tf.ones([n_dim, n_dim]), 0, 0) >= 1)
    L = entry_stop_gradients(L, L_mask)

    if not reverse:
        x = tf.transpose(x)
        x = tf.linalg.matmul(U,x)
        x = tf.linalg.matmul(L,x)
        x = tf.transpose(x)
    else:
        x = tf.transpose(x)
        x = tf.linalg.matmul(tf.linalg.inv(L), x)
        x = tf.linalg.matmul(tf.linalg.inv(U), x)
        x = tf.transpose(x)

    if logdet is not None:
        dlogdet = tf.reduce_sum(tf.math.log(tf.math.abs(U_diag)))
        if reverse:
            dlogdet *= -1.0
        return x, logdet + dlogdet

    return x

# actnorm layer: centering and scaling layer - simplification of batchnormalization
class actnorm(layers.Layer):
  def __init__(self, name, scale = 1.0, logscale_factor = 3.0, **kwargs):
    super(actnorm, self).__init__(name=name,**kwargs)
    self.scale = scale
    self.logscale_factor = logscale_factor

    self.data_init = True

  def build(self, input_shape):
    self.n_length = input_shape[-1]
    self.b     = self.add_weight(name='b', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)
    self.b_init = self.add_weight(name='b_init', shape=(1, self.n_length),
                                  initializer=tf.zeros_initializer(),
                                  dtype=tf.float32, trainable=False)

    self.logs  = self.add_weight(name='logs', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)
    self.logs_init = self.add_weight(name='logs_init', shape=(1, self.n_length),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=False)


  def call(self, inputs, logdet = None, reverse = False):
    # data initialization
    # by default, no data initialization is implemented.
    if not self.data_init:
        x_mean = tf.reduce_mean(inputs, [0], keepdims=True)
        x_var = tf.reduce_mean(tf.square(inputs-x_mean), [0], keepdims=True)

        self.b_init.assign(-x_mean)
        self.logs_init.assign(tf.math.log(self.scale/(tf.sqrt(x_var)+1e-6))/self.logscale_factor)

        self.data_init = True

    if not reverse:
      x = inputs + (self.b + self.b_init)
      x = x * tf.exp(self.logs + self.logs_init)
    else:
      x = inputs * tf.exp(-self.logs - self.logs_init)
      x = x - (self.b + self.b_init)

    if logdet is not None:
      dlogdet = tf.reduce_sum(self.logs + self.logs_init)
      if reverse:
        dlogdet *= -1
      return x, logdet + dlogdet

    return x

  def reset_data_initialization(self):
    self.data_init = False

# affine coupling layer: version 1, where we update the part II first.
class affine_coupling(layers.Layer):
  def __init__(self, name, n_split_at, n_width = 32, flow_coupling = 0, **kwargs):
    super(affine_coupling, self).__init__(name=name, **kwargs)
    # partition as [:n_split_at] and [n_split_at:]
    self.n_split_at = n_split_at
    self.flow_coupling = flow_coupling
    self.n_width = n_width

  def build(self, input_shape):
    n_length = input_shape[-1]
    if self.flow_coupling == 0:
      self.f = NN2('a2b', self.n_width, n_length-self.n_split_at)
    elif self.flow_coupling == 1:
      self.f = NN2('a2b', self.n_width, (n_length-self.n_split_at)*2)
    else:
      raise Exception()
    self.log_gamma  = self.add_weight(name='log_gamma',
                                      shape=(1, n_length-self.n_split_at),
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32, trainable=True)

  # the default setting performs a mapping of the data
  # without computing the jacobian
  def call(self, inputs, logdet=None, reverse=False):
    z = inputs
    n_split_at = self.n_split_at

    alpha=0.6

    if not reverse:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 += shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # orignal real NVP
        #scale = tf.nn.sigmoid(h[:,1::2]+2.0)
        #z2 += shift
        #z2 *= scale
        #if logdet is not None:
        #  logdet += tf.reduce_sum(tf.math.log(scale), axis=[1], keepdims=True)

        # resnet-like trick
        # we suppressed both the scale and the shift.
        scale = alpha*tf.nn.tanh(h[:,1::2])
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 = z2 + scale * z2 + shift
        if logdet is not None:
           logdet += tf.reduce_sum(tf.math.log(scale + tf.ones_like(scale)),
                                   axis=[1], keepdims=True)
      else:
        raise Exception()

      z = tf.concat([z1, z2], 1)
    else:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 -= shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # original real NVP
        #scale = tf.nn.sigmoid(h[:,1::2]+2.0)
        #z2 /= scale
        #z2 -= shift
        #if logdet is not None:
        #  logdet -= tf.reduce_sum(tf.math.log(scale), axis=[1], keepdims=True)

        # resnet-like trick
        # we suppressed both the scale and the shift.
        scale = alpha*tf.nn.tanh(h[:,1::2])
        shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        z2 = (z2 - shift) / (tf.ones_like(scale) + scale)
        if logdet is not None:
           logdet -= tf.reduce_sum(tf.math.log(scale + tf.ones_like(scale)),
                                   axis=[1], keepdims=True)
      else:
        raise Exception()

      z = tf.concat([z1, z2], 1)

    if logdet is not None:
        return z, logdet

    return z

# squeezing layer - KR rearrangement
class squeezing(layers.Layer):
    def __init__(self, name, n_dim, n_cut=1, **kwargs):
        super(squeezing, self).__init__(name=name, **kwargs)
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def call(self, inputs, reverse=False):
        z = inputs
        n_length = z.get_shape()[-1]

        if not reverse:
            if n_length < self.n_cut:
                raise Exception()

            if self.n_dim == n_length:
                if self.n_dim > 2 * self.n_cut:
                    if self.x is not None:
                        raise Exception()
                    else:
                        self.x = z[:, (n_length - self.n_cut):]
                        z = z[:, :(n_length - self.n_cut)]
                else:
                    self.x = None
            elif (n_length - self.n_cut) <= self.n_cut:
                z = tf.concat([z, self.x], 1)
                self.x = None
            else:
                cut = z[:, (n_length - self.n_cut):]
                self.x = tf.concat([cut, self.x], 1)
                z = z[:, :(n_length - self.n_cut)]
        else:
            if self.n_dim == n_length:
                n_start = self.n_dim % self.n_cut
                if n_start == 0:
                    n_start += self.n_cut
                self.x = z[:, n_start:]
                z = z[:, :n_start]

            x_length = self.x.get_shape()[-1]
            if x_length < self.n_cut:
                raise Exception()

            cut = self.x[:, :self.n_cut]
            z = tf.concat([z, cut], 1)

            if (x_length - self.n_cut) == 0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]
        return z

# one linear layer with defaul width 32.
class Linear(layers.Layer):
  def __init__(self, name, n_width=32, **kwargs):
    super(Linear, self).__init__(name=name, **kwargs)
    self.n_width = n_width

  def build(self, input_shape):
    n_length = input_shape[-1]
    self.w = self.add_weight(name='w', shape=(n_length, self.n_width),
                             initializer=default_initializer(),
                             dtype=tf.float32, trainable=True)
    self.b = self.add_weight(name='b', shape=(self.n_width,),
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32, trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

# two-hidden-layer neural network
class NN2(layers.Layer):
  def __init__(self, name, n_width=32, n_out=None, **kwargs):
    super(NN2, self).__init__(name=name, **kwargs)
    self.n_width = n_width
    self.n_out = n_out

  def build(self, input_shape):
    self.l_1 = Linear('h1', self.n_width)
    self.l_2 = Linear('h2', self.n_width)

    n_out = self.n_out or int(input_shape[-1])
    self.l_f = Linear('last', n_out)

  def call(self, inputs):
    # relu with low regularity
    x = tf.nn.relu(self.l_1(inputs))
    x = tf.nn.relu(self.l_2(x))

    # tanh with high regularity
    #x = tf.nn.tanh(self.l_1(inputs))
    #x = tf.nn.tanh(self.l_2(x))

    x = self.l_f(x)

    return x

# affine linear mapping from a bounded domain to [0,1]^d
class Affine_linear_mapping(layers.Layer):
    def __init__(self, name, lb, hb, **kwargs):
        super(Affine_linear_mapping, self).__init__(name=name, **kwargs)
        self.lb = lb
        self.hb = hb

    def call(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = (x - self.lb)/(self.hb - self.lb)
        else:
            x = (self.hb - self.lb)*x + self.lb

        if logdet is not None:
            dlogdet = tf.reduce_sum(tf.math.log(self.hb - self.lb))
            if not reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet
       
        return x

# Logistic mapping layer: mapping between a bounded domain (0,1)
#                         and an infinite domain (-inf, +inf)
# The default direction is from (-inf, +inf) to (0,1)
class Logistic_mapping(layers.Layer):
    def __init__(self, name, **kwargs):
        super(Logistic_mapping, self).__init__(name=name, **kwargs)
        self.s_init = 0.5

    def build(self, input_shape):
        n_length = input_shape[-1]
        self.s = self.add_weight(name='logistic_s', shape=(1, n_length),
                                 initializer = tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)

    # the direction of this mapping is not related to the flow
    # direction between the data and the prior
    def call(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = (tf.nn.tanh(x / (self.s_init + self.s)) + 1.0) / 2.0

            if logdet is not None:
                x = tf.clip_by_value(x, 1.0e-10, 1.0-1.0e-10)
                tp = tf.math.log(x) + tf.math.log(1-x) + tf.math.log(2.0/(self.s+self.s_init))
                dlogdet = tf.reduce_sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet
        else:
            x = tf.clip_by_value(x, 1.0e-10, 1.0-1.0e-10)
            tp1 = tf.math.log(x)
            tp2 = tf.math.log(1 - x)
            x = (self.s_init + self.s) / 2.0 * (tp1 - tp2)
            if logdet is not None:
                tp = tf.math.log((self.s+self.s_init)/2.0) - tp1 - tp2
                dlogdet = tf.reduce_sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet

        return x

