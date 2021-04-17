# Compact Bilinear Pooling implemented with TensorFlow2.0
This repository encompasses the TF2.0 implementation of Compact Bilinear Pooling using Random Maclaurin (RM) approximation method.
To understand more theoretical details, please refer to the paper of Yang Gao, et al. [1]

    
    class compact_bilinear_pooling(Model):
	"""
	A fast implementation of compact bilinear pooling layer/operation based on Random Maclaurin (RM) method [2] to approximate the polynomial kernel. 
	Build a compact bilinear pooling (CBP) layer to compute CBP results of convoluted feature maps.
	
    Args:
    	channel_num: 1st input, the channel number of feature maps produced by one upper layer
    	project_dim: 2nd input, the projection dimension which is a significant hyperparameter to tune for CBP, default to 8192 in [1].
    Returns:
    	x: compact bilinear pooled feature vectors with dimension d (equivalent to project_dim)
	"""
    	def __init__(self, channel_num, project_dim):

    	def call(self, inputs):
        
        return x


Reference: 

[1] Gao, Y., Beijbom, O., Zhang, N. and Darrell, T., 2016. Compact Bilinear Pooling. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

[2] Kar, P. and Karnick, H., 2012. Random Feature Maps for Dot Product Kernels. In: Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics. PMLR.
