import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class compact_bilinear_pooling(Model):
	"""
	A fast implementation of compact bilinear pooling [1] layer/operation based on Random Maclaurin (RM) method [2] to approximate the polynomial kernel. 
	Build a compact bilinear pooling (CBP) layer to compute CBP results of convoluted feature maps.
	Reference: 
    [1] Gao, Y., Beijbom, O., Zhang, N. and Darrell, T., 2016. Compact Bilinear Pooling. 
    In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.
    [2] Kar, P. and Karnick, H., 2012. Random Feature Maps for Dot Product Kernels. 
    In: Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics. PMLR.
    Args:
    	channel_num: 1st input, the channel number of feature maps produced by one upper layer
    	project_dim: 2nd input, the projection dimension which is a significant hyperparameter to tune for CBP, default to 8192 in [1].
    Returns:
    	x: compact bilinear pooled feature vectors with dimension d (equivalent to project_dim)
	"""
    
    def __init__(self, channel_num, project_dim=8192):
        super(compact_bilinear_pooling, self).__init__()
        self.w1 = np.random.randint(0, 2, (channel_num, project_dim))
        self.w1[self.w1 < 1] = -1
        self.w2 = np.random.randint(0, 2, (channel_num, project_dim))
        self.w2[self.w2 < 1] = -1
        self.d = tf.cast(project_dim, tf.float32)

    def call(self, inputs):
        a = tf.matmul(inputs, tf.convert_to_tensor(self.w1, dtype=tf.float32))
        b = tf.matmul(inputs, tf.convert_to_tensor(self.w2, dtype=tf.float32))
        x = tf.multiply(a, b) / tf.sqrt(self.d)
        x = tf.reduce_sum(x, [1, 2])  # sum pooling
        return x
