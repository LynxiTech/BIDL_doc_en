Model Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration file for the CIFAR10-DVS inner loop case is `resnetlif50-it-b16x1-cifar10dvs.py`, 
which uses the ResNet50 model structure, consistent with the outer loop model structure. 
The model backbone is named *ResNetLif*, and it is defined in the *bidlcls/models/backbones/residual/bidl_resnetlif.py* file.

Unlike the outer loop, regardless of running on an APU, the model accepts 5-dimensional data input [t,b,c,h,w]. 
For the Lif layer, the time axis is unfolded for computation, while for convolution layers and others, 
the time and batch dimensions are merged to be processed as 4-dimensional data.

Compilation and Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compiling on an APU, the Lif layer in the model structure needs to use the Lif2dIt class from the *bidlcls/models/layers/lif_itin.py* file.