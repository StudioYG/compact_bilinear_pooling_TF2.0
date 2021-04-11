# Compact Bilinear Pooling implemented using TF2.0
This repository contains the TF2.0 implementation of Compact Bilinear Pooling using Random Maclaurin (RM) approximation method.
To understand more theoretical details, please refer to the paper of Yang Gao, et al. [1]
class compact_bilinear_pooling(Model):
	"""
	A fast implementation of compact bilinear pooling layer/operation based on Random Maclaurin (RM) method to approximate the polynomial kernel. 
	Build a compact bilinear pooling (CBP) layer to compute CBP results of convoluted feature maps.
	Reference: [1] Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Args:
    	channel_num: 1st input, the channel number of feature maps produced by one upper layer
    	project_dim: 2nd input, the projection dimension which is a significant hyperparameter to tune for CBP, default to 8192 in [1].
    Returns:
    	x: compact bilinear pooled feature vectors with dimension d (equivalent to project_dim)
	"""
    
    def __init__(self, channel_num, project_dim):

    def call(self, inputs):
        
        return x
