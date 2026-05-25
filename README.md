
# Low-Complexity Channel Estimation for Spatial Non-Stationary XL-MIMO Systems: A Model-Based Deep Learning Approach
This is the code for [1]. 

[1] H. Lei, J. Zhang, Z. Liu, H. Xiao, B. Ai, D. W. K. Ng, and A. Nallanathan, “Low-Complexity Channel Estimation for Spatial Non-Stationary XL-MIMO Systems: A Model-Based Deep Learning Approach,” IEEE Transactions on Communications, to appear, 2026.


# Abstract
In this paper, we investigate the channel estimation problem in near-field extremely large-scale multiple-input multiple-output (XL-MIMO) systems, explicitly accounting for both spherical-wave propagation characteristics and spatial nonstationary effects. 
Building on these properties, we propose a novel model-based deep learning framework that delivers highaccuracy channel estimation with low computational complexity by tightly integrating domain knowledge and data-driven learning. 
Specifically, the proposed framework comprises three key unfolding networks: a sparse channel recovery network, a codebook update network, and an error cancellation network. 
The first network, referred to as variational Bayesian inference (VBI)-Net, is derived by unfolding the inverse-free VBI (IF-VBI) algorithm.
It enables high-precision sparse channel reconstruction without requiring explicit prior assumptions, by learning the underlying precision distribution directly from data. 
The second network, gradient (Grad)-Net, is developed by unfolding the gradient ascent procedure, where learnable step sizes are introduced to adaptively refine the parameters of the polar-domain grids. 
Moreover, Grad-Net captures spatial non-stationary characteristics associated with the polar-domain representation by jointly exploiting gradient information and estimated path parameters. 
The third network, termed projected gradient descent (PGD)-Net, is constructed by unfolding the PGD algorithm. It iteratively refines the channel estimates and effectively suppresses residual estimation errors induced by spherical-wave propagation and spatial non-stationarity. 
Extensive numerical simulations demonstrate that the proposed framework significantly outperforms existing methods in both estimation accuracy and computational efficiency. 
Furthermore, the proposed framework achieves a superior accuracy-complexity tradeoff for practical XL-MIMO systems, delivering enhanced performance while maintaining very low computational complexity.

# Index Terms
Near-field, XL-MIMO, channel estimation, deep unfolding, spatial non-stationary.

# License and Referencing
If you in any way use this code for research that results in publications, please cite our original article listed above ([1]).

[1] H. Lei, J. Zhang, Z. Liu, H. Xiao, B. Ai, D. W. K. Ng, and A. Nallanathan, “Low-Complexity Channel Estimation for Spatial Non-Stationary XL-MIMO Systems: A Model-Based Deep Learning Approach,” IEEE Transactions on Communications, to appear, 2026.
