from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

# a wrapper for tensorflow.dataset
class dataflow(object):
    def __init__(self, x, buffersize, batchsize, y=None):
        self.x = x
        self.y = y
        self.buffersize = buffersize
        self.batchsize = batchsize

        if y is not None:
            dx = tf.data.Dataset.from_tensor_slices(x)
            dy = tf.data.Dataset.from_tensor_slices(y)
            self.dataset = tf.data.Dataset.zip((dx, dy))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(x)

        self.batched_dataset = self.dataset.batch(batchsize)
        self.shuffled_batched_dataset = self.dataset.shuffle(buffersize).batch(batchsize)

    def get_shuffled_batched_dataset(self):
        return self.shuffled_batched_dataset

    def get_batched_dataset(self):
        return self.batched_dataset

    def update_shuffled_batched_dataset(self):
        self.shuffled_batched_dataset = self.dataset.shuffle(self.buffersize).batch(self.batchsize)
        return self.shuffled_batched_dataset

    def get_n_batch_from_shuffled_batched_dataset(self, n):
        it = iter(self.shuffled_batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

    def get_n_batch_from_batched_dataset(self, n):
        it = iter(self.batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

# when threshold > 0, it returns a Gaussian with an ellipse hole
def gen_2d_Gaussian_w_hole(ndim, n_train, alpha, theta, threshold, weighted=False):
    assert ndim == 2
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs = np.zeros((ndim,ndim), dtype='float32')
    Rs[0,0] = alpha * np.cos(theta)
    Rs[0,1] = -alpha * np.sin(theta)
    Rs[1,0] = np.sin(theta)
    Rs[1,1] = np.cos(theta)
 
    #tf.random.set_seed(12345)
    for i in range(m):
        while True:
            z = np.random.normal(0,1,ndim)
            y = np.matmul(Rs,z)
            if np.linalg.norm(y,2) >= threshold:
                x[i,:] = z
                break

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        beta = 0.5
        for i in range(m):
            y = np.matmul(Rs,x[i,:])
            w[i,0] = np.exp(beta*(np.linalg.norm(y,2)-threshold))

        s = np.sum(w)
        for i in range(m):
            w[i,0] = w[i,0]*m/s
        
        return x,w

    return x


def gen_2d_Logistic_w_hole(ndim, n_train, scale, alpha, theta, threshold, weighted=False):
    assert ndim == 2
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs = np.zeros((ndim,ndim), dtype='float32')
    Rs[0,0] = alpha * np.cos(theta)
    Rs[0,1] = -alpha * np.sin(theta)
    Rs[1,0] = np.sin(theta)
    Rs[1,1] = np.cos(theta)

    #tf.random.set_seed(12345)
    for i in range(m):
        while True:
            z = np.random.logistic(0,scale,ndim)
            y = np.matmul(Rs,z)
            if np.linalg.norm(y,2) >= threshold:
                x[i,:] = z
                break

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        beta = 0.5
        for i in range(m):
            y = np.matmul(Rs,x[i,:])
            w[i,0] = np.exp(beta*(np.linalg.norm(y,2)-threshold))

        s = np.sum(w)
        for i in range(m):
            w[i,0] = w[i,0]*m/s

        return x,w

    return x

def gen_xd_Logistic_w_2d_hole(ndim, n_train, scale, alpha, theta, threshold, weighted=False):
    m  = n_train

    x = np.zeros((m,ndim), dtype='float32')

    Rs_even = np.zeros((2,2), dtype='float32')
    Rs_even[0,0] = alpha * np.cos(theta)
    Rs_even[0,1] = -alpha * np.sin(theta)
    Rs_even[1,0] = np.sin(theta)
    Rs_even[1,1] = np.cos(theta)

    Rs_odd = np.zeros((2,2), dtype='float32')
    Rs_odd[0,0] = alpha * np.cos(np.pi-theta)
    Rs_odd[0,1] = -alpha * np.sin(np.pi-theta)
    Rs_odd[1,0] = np.sin(np.pi-theta)
    Rs_odd[1,1] = np.cos(np.pi-theta)

    n = 0.0
    y = np.zeros((ndim-1,), dtype='float32')
    for i in range(m):
        while True:
            z = np.random.logistic(0, scale, ndim)
            n += 1.0

            for j in range(ndim-1):
                if j % 2 == 0:
                    tp1 = Rs_even[0,0]*z[j] + Rs_even[0,1]*z[j+1]
                    tp2 = Rs_even[1,0]*z[j] + Rs_even[1,1]*z[j+1]
                else:
                    tp1 = Rs_odd[0,0]*z[j] + Rs_odd[0,1]*z[j+1]
                    tp2 = Rs_odd[1,0]*z[j] + Rs_odd[1,1]*z[j+1]

                y[j] = np.sqrt(tp1**2+tp2**2)
            if np.amin(y) >= threshold:
                x[i,:] = z

                if (i+1) % 10000 == 0:
                    print("step {}:".format(i+1))

                break

    return x, float(m)/n

def gen_4d_Gaussian_w_hole(ndim, n_train, alpha, theta, threshold, weighted=False, loadfile=False):
    assert ndim == 4
    m = n_train

    x = np.zeros((m,ndim), dtype='float32')

    if loadfile is True:
        x = np.loadtxt('training_set_KR.dat').astype(np.float32)
    else:
        Rs = np.zeros((2,2), dtype='float32')
        Rs[0,0] = alpha * np.cos(theta)
        Rs[0,1] = -alpha * np.sin(theta)
        Rs[1,0] = np.sin(theta)
        Rs[1,1] = np.cos(theta)

        for i in range(m):
            while True:
                z = np.random.normal(0,1,2)
                y = np.matmul(Rs,z)
                if np.linalg.norm(y,2) >= threshold:
                    x[i,:2] = z
                    break

        Rs[0,0] = alpha * np.cos(theta+np.pi/2.0)
        Rs[0,1] = -alpha * np.sin(theta+np.pi/2.0)
        Rs[1,0] = np.sin(theta+np.pi/2.0)
        Rs[1,1] = np.cos(theta+np.pi/2.0)

        for i in range(m):
            while True:
                z = np.random.normal(0,1,2)
                y = np.matmul(Rs,z)
                if np.linalg.norm(y,2) >= threshold:
                    x[i,2:] = z
                    break


        np.savetxt('training_set_KR.dat'.format(), x)

    if weighted is True:
        w = np.ones((m,1), dtype='float32')
        return x,w

    return x

def gen_nd_Gaussian_w_hole(ndim, n_train, threshold):
    m = n_train

    x = np.zeros((m,ndim), dtype='float32')
    for i in range(m):
        while True:
            z = np.random.normal(0,1,ndim)
            if np.linalg.norm(z,2) > threshold:
                x[i,:] = z
                break
    return x

# for random variables:
def flatten_sum(logps):
    assert len(logps.get_shape()) == 2
    return tf.reduce_sum(logps, [1], keepdims=True)

def gaussian_diag():
    class o(object): pass
    o.logps = lambda x: -0.5*(tf.math.log(2.*np.pi)+x**2)
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.logps_g = lambda x, mean, logsd: -0.5*(tf.math.log(2.*np.pi)+2.*logsd+(x-mean)**2/tf.exp(2.*logsd))
    o.logp_g  = lambda x, mean, logsd: flatten_sum(o.logps_g(x, mean, logsd))
    return o

# Gaussian distribution
def log_standard_Gaussian(x):
    return flatten_sum(-0.5*(tf.math.log(2.*np.pi)+x**2))

# logistic distribution
def log_logistic(x):
    s = 2.0
    return flatten_sum(-x/s-tf.math.log(s)-2.0*tf.math.log(1.0+tf.exp(-x/s)))

