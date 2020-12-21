from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import BR_lib.BR_layers as BR_layers
import BR_lib.BR_data as BR_data

# invertible mapping based on real NVP
class IM_rNVP(tf.keras.Model):
    def __init__(self, name, lb, hb, n_depth, n_split_at, width=32, flow_coupling=0,
	         rotation=False, bounded_supp=False, **kwargs):
        super(IM_rNVP, self).__init__(name=name,**kwargs)
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.width = width
        self.flow_coupling = flow_coupling
        self.rotation = rotation
        self.bounded_supp = bounded_supp
        self.lb = lb
        self.hb = hb

	# two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0

        # add a logistic mapping to refine the data in a bounded support
        if bounded_supp:
            self.affine_linear = BR_layers.Affine_linear_mapping('affine_linear', self.lb, self.hb)
            self.logistic_mapping = BR_layers.Logistic_mapping('logi_mapping')

        # a rotation layer
        if rotation:
            self.W_LU = BR_layers.W_LU('w_lu')

        # stack affine coupling layers
        self.flow_mapping = BR_layers.flow_mapping('flow_mapping',
                                                   n_depth,
                                                   n_split_at,
                                                   width=width,
                                                   flow_coupling=flow_coupling)

        # the prior distribution
        self.log_prior = BR_data.log_standard_Gaussian
        #self.log_prior = BR_data.log_logistic

    # computing the logarithm of the estimated pdf on the input data.
    def call(self, inputs):
        objective = tf.zeros_like(inputs, dtype=tf.float32)[:,0]
        objective = tf.reshape(objective, [-1,1])

        z = inputs
        
        if self.bounded_supp:
            z, objective = self.affine_linear(z, objective)
            z, objective = self.logistic_mapping(z, objective, reverse=True)

        # unitary transformation
        if self.rotation:
            z, objective = self.W_LU(z, objective)

        # f(y) and log of jacobian
        z, objective = self.flow_mapping(z, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs
        if logdet is not None:
            if self.bounded_supp:
                z, logdet = self.affine_linear(z, logdet)
                z, logdet = self.logistic_mapping(z, logdet, reverse=True)
            if self.rotation:
                z, logdet = self.W_LU(z, logdet)
            z, logdet = self.flow_mapping(z, logdet)
            return z, logdet
        else:
            if self.bounded_supp:
                z = self.affine_linear(z)
                z = self.logistic_mapping(z, reverse=True)
            if self.rotation:
                z = self.W_LU(z)
            z = self.flow_mapping(z)
            return z

    # mapping from prior to data subject to the estimated distribution
    def mapping_from_prior(self, inputs):
        z = self.flow_mapping(inputs, reverse=True)
        if self.rotation:
            z = self.W_LU(z, reverse=True)
        if self.bounded_supp:
            z = self.logistic_mapping(z)
            z = self.affine_linear(z, reverse=True)
        return z

    def actnorm_data_initialization(self):
        self.flow_mapping.actnorm_data_initialization()

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return tf.random.normal((n_samples, n_dim))

# invertible mapping based on real NVP and KR rearrangement
class IM_rNVP_KR(tf.keras.Model):
    def __init__(self, name, n_dim, lb, hb, n_step, n_depth,
                 n_width=32, shrink_rate=1.0,
                 flow_coupling=0, rotation=False,
                 bounded_supp=False, **kwargs):
       super(IM_rNVP_KR, self).__init__(name=name,**kwargs)

       # two affine coupling layers are needed for each update of the vector
       assert n_depth % 2 == 0

       self.n_dim = n_dim # dimension of the data
       self.n_step = n_step # step size for dimension reduction
       self.n_depth = n_depth # depth for flow_mapping
       self.n_width = n_width
       self.shrink_rate = shrink_rate
       self.flow_coupling = flow_coupling
       self.rotation = rotation
       self.bounded_supp = bounded_supp
       self.lb = lb
       self.hb = hb

       # the number of filtering stages
       self.n_stage = n_dim // n_step
       if n_dim % n_step == 0:
           self.n_stage -= 1

       if bounded_supp:
           self.affine_linear = BR_layers.Affine_linear_mapping('affine_linear', self.lb, self.hb)
           self.logistic_mapping = BR_layers.Logistic_mapping('logi_mapping')

       # the number of rotation layers
       self.n_rotation = self.n_stage

       if rotation:
           self.rotations = []
           for i in range(self.n_rotation):
               # rotate the coordinate system for a better representation of data
               self.rotations.append(BR_layers.W_LU('rotation'+str(i)))

       # flow mapping with n_stage
       self.flow_mappings = []
       for i in range(self.n_stage):
           # flow_mapping given by such as real NVP
           n_split_at = n_dim - (i+1) * n_step
           self.flow_mappings.append(BR_layers.flow_mapping('flow_mapping'+str(i),
                                     n_depth, 
		 	                         n_split_at,
                                     n_width=n_width, 
                                     flow_coupling=flow_coupling))
           n_width = int(n_width*self.shrink_rate)

       # data will pass the squeezing layer at the end of each stage
       self.squeezing_layer = BR_layers.squeezing('squeezing', n_dim, n_step)

       # the prior distribution is the standard Gaussian
       self.log_prior = BR_data.log_standard_Gaussian

    # computing the logarithm of the estimated pdf on the input data.
    def call(self, inputs):
        objective = tf.zeros_like(inputs, dtype='float32')[:,0]
        objective = tf.reshape(objective, [-1,1])

        # f(y) and log of jacobian
        z, objective = self.mapping_to_prior(inputs, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs
        if self.bounded_supp:
            z = self.affine_linear(z, logdet)
            if logdet is not None:
                z, logdet = z
            z = self.logistic_mapping(z, logdet, reverse=True)
            if logdet is not None:
                z, logdet = z

        for i in range(self.n_stage):
            if logdet is not None:
                if self.rotation and i < self.n_rotation:
                    z, logdet = self.rotations[i](z, logdet)
                z, logdet = self.flow_mappings[i](z, logdet)
            else:
                if self.rotation and i < self.n_rotation:
                    z = self.rotations[i](z)
                z = self.flow_mappings[i](z)

            z = self.squeezing_layer(z)

        if logdet is not None:
            return z, logdet
        else:
            return z

    # mapping from prior to data
    def mapping_from_prior(self, inputs):
        z = inputs
        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer(z, reverse=True)
            z = self.flow_mappings[i](z, reverse=True)
            if self.rotation and i < self.n_rotation:
                z = self.rotations[i](z, reverse=True)
        if self.bounded_supp:
            z = self.logistic_mapping(z)
            z = self.affine_linear(z, reverse=True)

        return z

    # data initialization for actnorm layers
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return tf.random.normal((n_samples, n_dim))

