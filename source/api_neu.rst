API of Neuronal Models
===============================================================================================

Main Locations of LIF/LIFPlus Modules:

- *.layers/lif.py*
- *.layers/lifplus.py*

These LIF/LIFPlus modules primarily provide a meta-model class Lif inherited from *torch.nn.Module*, and various classes divided by dimensions 1d and 2d, wrapped by the iterable It class for the timeline.

.. _sjymx:

Neuronal Models
-----------------------------------------------------------------------------------------------

- Lif/LifPlus

  - 1d Processing One-dimensional Data

    - Leak Integrate-and-Fire Model with Fully Connected Synapses FcLif/FcLifPlus
    - Leak Integrate-and-Fire Model Lif1d/LifPlus1d

  - 2d Processing Two-dimensional Data

    - Leak Integrate-and-Fire Model with Convolutional Synapses Conv2dLif/Conv2dLifPlus
    - Leak Integrate-and-Fire Model Lif2d/LifPlus2d

.. _lif-lifplus:

Basic Neuron Models——Lif/LifPlus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inheriting from torch.nn.Module, the forward method in this class implements the complete process of LIF/LIFPlus neuron computation.

This class depicts the most basic and typical operator formula in LIF/LIFPlus, and its methods are mainly inherited by the classes introduced later.

Initialization/Adjustable parameters of Lif/LifPlus can be found in the following table.

Table: Initialization/Adjustable Parameters of Lif

+----------------+------------------------------------------------------------+
| Parameter      | Description                                                |
+================+============================================================+
| norm           | BatchNorm object, callable type.                           |
+----------------+------------------------------------------------------------+
| mode           | Desired output type.                                       |
|                |                                                            |
|                | [Data Type] STR type, with two types.                      |
|                |                                                            |
|                | [Value Range]                                              |
|                |                                                            |
|                | - spike: Corresponds to LIF model, expected output {0,1}   |
|                |   spikes;                                                  |
|                | - analog: Corresponds to LIAF model, expected output analog|
|                |   signals.                                                 |
+----------------+------------------------------------------------------------+
| memb_mode      | Membrane potential related parameters.                     |
|                |                                                            |
|                | [Data Type] tuple                                          |
|                |                                                            |
|                | For detailed explanation see :ref:`memb_mode`              |
+----------------+------------------------------------------------------------+
| soma_params    | Key parameters of neuron cell body.                        |
|                |                                                            |
|                | [Data Type] Dict                                           |
|                |                                                            |
|                | For detailed explanation see :ref:`soma_params`            |
+----------------+------------------------------------------------------------+
| noise          | Noise, float type.                                         |
+----------------+------------------------------------------------------------+
| ON_APU         | Chip inference flag, indicating whether inference is       |
|                | performed on the KA200 chip.                               |
|                | [Data Type] bool type                                      |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - True: Inference on KA200;                                |
|                | - False: Inference on Nvidia GPU.                          |
+----------------+------------------------------------------------------------+
| spike_func     | [Data Type] callable                                       |
|                |                                                            |
|                | Gradient replacement function used by neurons              |
+----------------+------------------------------------------------------------+
| use_inner_loop | [Data Type] bool type                                      |
|                |                                                            |
|                | Whether to use the inner loop mode of neurons              |
+----------------+------------------------------------------------------------+

Table: Initialization/Adjustable Parameters of LifPlus

+----------------+------------------------------------------------------------+
| Parameter      | Description                                                |
+================+============================================================+
| norm           | BatchNorm object, callable type.                           |
+----------------+------------------------------------------------------------+
| mode           | Desired output type.                                       |
|                |                                                            |
|                | [Data Type] STR type, with two types.                      |
|                |                                                            |
|                | [Value Range]                                              |
|                |                                                            |
|                | - spike: Corresponds to LIF model, expected output {0,1}   |
|                |   spikes;                                                  |
|                | - analog: Corresponds to LIAF model, expected output analog| 
|                |   signals.                                                 |
+----------------+------------------------------------------------------------+
| memb_mode      | Membrane potential related parameters.                     |
|                |                                                            |
|                | [Data Type] tuple                                          |
|                |                                                            |
|                | For detailed explanation see :ref:`memb_mode`              |
+----------------+------------------------------------------------------------+
| soma_params    | Key parameters of neuron cell body.                        |
|                |                                                            |
|                | [Data Type] Dict                                           |
|                |                                                            |
|                | For detailed explanation see: :ref:`soma_params`           |
+----------------+------------------------------------------------------------+
| input_accm     | Input spike accumulation related parameters.               |
|                |                                                            |
|                | [Data Type] int                                            |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - 0: CUB (current accumulation);                           |
|                | - 1: COBE (exponential function conductance accumulation); |
|                | - 2: COBA (alpha function conductance accumulation).       |
+----------------+------------------------------------------------------------+
| rev_volt       | Input spike accumulation related parameters.               |
|                |                                                            |
|                | [Data Type] bool                                           |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - True: with REV (set reversal voltage);                   |
|                | - False: without REV (do not set reversal voltage).        |
+----------------+------------------------------------------------------------+
| fire_refrac    | Refractory period related parameters.                      |
|                |                                                            |
|                | [Data Type] int                                            |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - 0: no refractory (not set);                              |
|                | - 1: AR (absolute refractory period).                      |
+----------------+------------------------------------------------------------+
| spike_init     | Spike initiation related parameters.                       |
|                |                                                            |
|                | [Data Type] int                                            |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - 0: naïve (same as LIF);                                  |
|                | - 1: EXI (exponential spike initiation);                   |
|                | - 2: QDI (quadratic spike initiation).                     |
+----------------+------------------------------------------------------------+
| trig_current   | Spike trigger current related parameters.                  |
|                |                                                            |
|                | [Data Type] int                                            |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - 0: naïve (same as LIF);                                  |
|                | - 1: ADT (adaptation);                                     |
|                | - 2: SBT (subthreshold oscillation).                       |
+----------------+------------------------------------------------------------+
| memb_decay     | Membrane decay related parameters.                         |
|                |                                                            |
|                | [Data Type] int                                            |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - 0: naïve (same as LIF);                                  |
|                | - 1: EXD (exponential decay).                              |
+----------------+------------------------------------------------------------+
| noise          | Noise, float type.                                         |
+----------------+------------------------------------------------------------+
| ON_APU         | Chip inference flag, indicating whether inference is       |
|                | performed on the KA200 chip.                               |
|                |                                                            |
|                | [Data Type] bool type                                      |
|                |                                                            |
|                | [Value Type]                                               |
|                |                                                            |
|                | - True: Inference on KA200;                                |
|                | - False: Inference on Nvidia GPU.                          |
+----------------+------------------------------------------------------------+
| spike_func     | [Data Type] callable                                       |
|                |                                                            |
|                | Gradient replacement function used by neurons              |
+----------------+------------------------------------------------------------+
| use_inner_loop | [Data Type] bool type                                      |
|                |                                                            |
|                | Whether to use the inner loop mode of neurons              |
+----------------+------------------------------------------------------------+

.. _memb_mode:

memb_mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

memb_mode represents membrane potential related parameters. The data type is a tuple, containing two elements:

+---------+------------------------------------------------------------+
| Parameter| Description                                               |
+=========+============================================================+
| First   | Represents the four different states of membrane potential |
| Element | experienced from the start of the previous signal input in |
|         | LIF or LIAF models, including: membrane potential after    |
|         | receiving stimulus; membrane potential after subtracting   |
|         | firing threshold; reset membrane potential; membrane       |
|         | potential after leak. These four states correspond to      |
|         | different membrane potential states expected by LIF or     |
|         | LIAF models.                                               |
|         |                                                            |
|         | [Data Type] int type                                       |
|         |                                                            |
|         | [Value Range]                                              |
|         |                                                            |
|         | - 0: membrane potential after input;                       |
|         | - 1: membrane potential after subtracting firing threshold;|
|         | - 2: reset membrane potential;                             |
|         | - 3: membrane potential after leak.                        |
+---------+------------------------------------------------------------+
| Second  | Activation function type set for the membrane potential    |
| Element | obtained from the first element.                           |
|         |                                                            |
|         | [Data Type] callable type                                  |
|         |                                                            |
|         | [Configuration Description] The function of this activation|
|         | function is to convert the pulse signal {0,1} transmitted  |
|         | in the model into an analog signal through an activation   |
|         | function, such as torch.relu, torch.sigmoid, or torch.tanh,|
|         | etc. If the value is None, no activation function is added.|
|         | This element is only valid when mode is set to analog.     |
+---------+------------------------------------------------------------+

.. _soma_params:

soma_params(Lif)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

soma_params represent key parameters of neuron cell bodies.

[Default Parameters]

::

   SOMA_PARAMS = {
      'alpha': .3,
      'beta': 0.,
      'theta': .5,
      'v_0': 0.,
      'shape': [],
      'learn': False
   }

[Parameter Description]

+---------+------------------------------------------------------------+
| Parameter| Description                                               |
+=========+============================================================+
| alpha   | Parameters of the cell body used to compute cell membrane  |
|         | potential values in different states before and after pulse|
|         | firing.                                                    |
|         |                                                            |
| beta    | [Data Type] float type                                     |
|         |                                                            |
| theta   | [Configuration Description]                                |
|         |                                                            |
| v_0     | - alpha: multiplicative leak.                              |
|         | - beta: additive leak.                                     |
|         | - theta: threshold.                                        |
|         | - v_0: reset membrane potential.                           |
+---------+------------------------------------------------------------+
| shape   | Indicates the sharing degree of cell body parameters.      |
|         |                                                            |
|         | [Data Type] list type                                      |
|         |                                                            |
|         | [Value Range]                                              |
|         |                                                            |
|         | Determines whether all neurons share a set of parameters or|
|         | each channel has a separate parameter based on the shape of|
|         | ``shape``:                                                 |
|         |                                                            |
|         | - If ``shape`` is an empty ``[]``, it indicates that all   |
|         |   neurons share one set of parameters, corresponding to    |
|         |   ``soma_params`` value of ``all_share`` in the config     |
|         |   file;                                                    |
|         |                                                            |
|         | - If each channel has a separate parameter, assuming the   |
|         |   number of channels is c, for the fully connected version,|
|         |   ``shape`` should be set to ``[1,c]``, while for the      |
|         |   convolutional version, ``shape`` should be set to        |
|         |   ``[1, c, 1,1]``, corresponding to ``channel_share``      |
|         |   in the config file.                                      |
|         |                                                            |
|         | [Configuration Description]                                |
|         |                                                            |
|         | Method to set the ``soma_params`` value in the config file |
|         | see :ref:`config_content`.                                 |
+---------+------------------------------------------------------------+
| learn   | Whether to obtain the key parameters of the cell body      |
|         | through learning.                                          |
|         |                                                            |
|         | [Data Type] bool type                                      |
|         |                                                            |
|         | [Value Range] False, True                                  |
|         |                                                            |
|         | - False: fixed parameters (not obtained through learning). |
|         | - True: update relevant parameters through learning.       |
|         |                                                            |
|         | [Default Value] False                                      |
|         |                                                            |
|         | [Configuration Description] Currently, configuration       |
|         | modification is not supported.                             |
+---------+------------------------------------------------------------+

soma_params (LifPlus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`soma_params` represents the key parameters of the neuron's soma.

[Default Parameters]

::

   SOMA_PARAMS = {
      'epsilon': None,
      'v_g': None,
      'tau_recip': None,
      'v_0': 0.,
      'epsilon_r': None,
      'v_rr': None,
      'v_ar': None,
      'q_r': None,
      'b': None,
      'epsilon_w': None,
      'theta': .5,
      'v_theta': None,
      'delta_t': None,
      'v_c': None,
      'a': None,
      'v_w': None,
      'alpha': .3,
      'beta': 0.,
      'v_leak': None,
      'shape': [],
      'learn': False,
   }

[Parameter Description]

+------------+-------------------------------------------------------------+
| Parameter  | Description                                                 |
+============+=============================================================+
| epsilon    | Soma parameter for calculating membrane potential values in |
|            | different states before and after spiking.                  |
|            |                                                             |
| v_g        | [Data Type] float                                           |
|            |                                                             |
| tau_recip  | [Configuration Description]                                 |
|            |                                                             |
| v_0        | - epsilon: Conductance decay constant.                      |
|            |                                                             |
| epsilon_r  | - v_g: Reverse voltage constant.                            |
|            |                                                             |
| v_rr       | - tau_recip: ∆𝑡/𝜏  (∆𝑡 represents the sampling interval,    |
|            |   𝜏 as the neuron time constant).                           |
|            |                                                             |
| v_ar       | - v_0: Reset membrane potential.                            |
|            |                                                             |
| q_r        | - epsilon_r: Relative refractory decay constant.            |
|            |                                                             |
| b          | - v_rr: Relative refractory turn voltage.                   |
|            |                                                             |
| epsilon_w  | - v_ar: Adaptive reverse voltage.                           |
|            |                                                             |
| theta      | - q_r: Relative refractory jump size.                       |
|            |                                                             |
| v_theta    | - b: Spike-triggered jump size.                             |
|            |                                                             |
| delta_t    | - epsilon_w: Adaptive decay constant.                       |
|            |                                                             |
| v_c        | - theta: Threshold.                                         |
|            |                                                             |
| a          | - v_theta: Trigger voltage.                                 |
|            |                                                             |
| v_w        | - delta_t: Sharpness factor.                                |
|            |                                                             |
| alpha      | - v_c: Critical voltage.                                    |
|            |                                                             |
| beta       | - a: Subthreshold coupling constant.                        |
|            |                                                             |
| v_leak     | - v_w: Coupling membrane potential offset constant.         |
|            |                                                             |
|            | - alpha: Multiplicative leakage.                            |
|            |                                                             |
|            | - beta: Additive leakage.                                   |
|            |                                                             |
|            | - v_leak: Linear delay constant.                            |
+------------+-------------------------------------------------------------+
| shape      | Indicates the degree of sharing of soma parameters.         |
|            |                                                             |
|            | [Data Type] list                                            |
|            |                                                             |
|            | [Value Range]                                               |
|            |                                                             |
|            | Determine whether all neurons share a set of parameters or  |
|            | each channel has separate parameters based on the shape:    |
|            |                                                             |
|            | - If shape is [], i.e., an empty list, it means all neurons |
|            |   share the same parameters, corresponding to the value of  |
|            |   ``soma_params`` in the configuration file being           |
|            |   ``all_share``.                                            |
|            |                                                             |
|            | - If each channel has separate parameters, assuming the     |
|            |   number of channels is c, for the fully connected version, |
|            |   set ``shape`` to ``[1,c]``, and for the convolutional     |
|            |   version, set ``shape`` to ``[1, c, 1,1]``, corresponding  |
|            |   to the value of ``soma_params`` in the configuration file |
|            |   being ``channel_share``.                                  |
|            |                                                             |
|            | [Configuration Description]                                 |
|            |                                                             |
|            | Method to set ``soma_params`` values in the configuration   |
|            | file can be seen in :ref:`config_content`.                  |
+------------+-------------------------------------------------------------+
| learn      | Indicates whether the key parameters of the soma are        |
|            | obtained through learning.                                  |
|            |                                                             |
|            | [Data Type] bool                                            |
|            |                                                             |
|            | [Value Range] False, True                                   |
|            |                                                             |
|            | - False: Fixed parameters (not learned).                    |
|            | - True: Learned and updated parameters through all layers.  |
|            |                                                             |
|            | [Default Value] False                                       |
|            |                                                             |
|            | [Configuration Description] Configuration change is         |
|            | currently unsupported.                                      |
+------------+-------------------------------------------------------------+

Hardware-accelerated Functions in Lif/LifPlus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lif includes two hardware-accelerated functions that can execute swiftly on Lynxi chip, defined as follows:

1. `cmpandfire`: Compare and fire function, defined as:

   .. math:: y(i)\  = \ 1\ if\ x(i)\  > \ theta,\ otherwise\ 0

   where x, y are Tensors, th is a scalar, i represents any element index in the Tensor.

2. `resetwithdecay`: Calculate reset and leakage after firing, used for resetting and leakage calculation process after firing, defined as:

   .. math:: y(i) = \ alpha\ *\ v\_ 0\  + \ beta\ if\ x(i)\  > \ theta，otherwise\ alpha\ *\ x(i)\  + \ beta

   where x, y are tensors; th, decay, reset are scalars.

These functions are expressed using PyTorch custom layers and conveyed to Lynxi compiler (though not an execution description by the actual Lynxi chip), and they can also be used in the portrayal of layers outside of Lif/LifPlus.

State Access and Description in Lif/LifPlus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the Lyngor compiler only compiles single-time-shot computation logic, the default logic is that after computing for the current time shot, all variable lifecycles end and are not saved. However, for neurons, certain state information needs to be retained for subsequent use in the next time shot, such as membrane potential and postsynaptic currents. Thus, auxiliary code is needed to identify which variables need to be statically stored and which variables need to be retrieved from static storage during the current calculation. Therefore, the following convention is established:

For a state variable `v` (such as membrane potential), before its first usage in the current time shot, i.e., before `v` first appears on the right-hand side of a computation equation, you need to add:

::

   load(v, 'v_string')

where `v` denotes the variable name of the state variable, ``v_string`` is a globally unique string identifier, which can be arbitrarily assigned; it’s currently generated using UUID, but the same variable's ``v_string`` must be consistent.

Typically, loading operations for a series of state variables can be collectively placed at the beginning of the neuron model description.

Similarly, after the final update of the state information `v` in the current time shot, i.e., after the last appearance on the left-hand side of a computation equation, you need to add:

::

   save(v, 'v_string')

Typically, saving operations for a series of state variables can be collectively placed at the end of the neuron model description.

.. attention::

   For a state variable, `load/save` must be paired.

One-dimensional Data Processing Model with Fully Connected Synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FcLif
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherits from ``torch.nn.Module``, FcLif1d is based on Lif. In the ``init`` method, initial values are assigned to the parameters involved in Lif. It can only process a single time step input during use.

The initialization/adjustable parameters for the FcLif layer are listed in the table below.

Table: FcLif Layer Structure Parameter Description

+----------------+-------------------------------------------------------+
| Parameter      | **Meaning**                                           |
+================+=======================================================+
| input_channel  | Number of input layer channels, int type.             |
+----------------+-------------------------------------------------------+
| hidden_channel | Number of hidden or output layer channels, int type.  |
+----------------+-------------------------------------------------------+
| feed_back      | Whether to use feedback loops linking hidden layers,  |
|                | bool type, default is ``False``.                      |
+----------------+-------------------------------------------------------+
| norm_state     | Whether to use BatchNorm for normalization to prevent |
|                | gradient explosion, bool type, default is ``False``.  |
+----------------+-------------------------------------------------------+
| mode           | Specific contents are the same as :ref:`lif-lifplus`  |
|                | chapter's Table Initialization/Adjustable Parameters  |
| memb_mode      | of Lif.                                               |
|                |                                                       |
| soma_params    |                                                       |
|                |                                                       |
| noise          |                                                       |
+----------------+-------------------------------------------------------+
| spike_func     | [Data Type] callable                                  |
|                |                                                       |
|                | Gradient surrogate function used by neurons           |
+----------------+-------------------------------------------------------+
| use_inner_loop | [Data Type] bool type                                 |
|                |                                                       |
|                | Whether to use neuron inner loop mode                 |
+----------------+-------------------------------------------------------+
| it_batch       | [Data Type] int type                                  |
|                |                                                       |
|                | Inner loop mode's ``batch_size``, use ``1`` during    |
|                | APU inference, can be defined freely during training  |
+----------------+-------------------------------------------------------+

FcLifPlus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherits from ``torch.nn.Module``, FcLifPlus1d is based on LifPlus. In the ``init`` method, initial values are assigned to the parameters involved in LifPlus. It can only process a single time step input during use.

The initialization/adjustable parameters for the FcLifPlus layer are listed in the table below.

Table: FcLif Layer Structure Parameter Description

+----------------+-------------------------------------------------------+
| **Parameter**  | **Meaning**                                           |
+================+=======================================================+
| input_channel  | Number of input layer channels, int type.             |
+----------------+-------------------------------------------------------+
| hidden_channel | Number of hidden or output layer channels, int type.  |
+----------------+-------------------------------------------------------+
| feed_back      | Whether to use feedback loops linking hidden layers,  |
|                | bool type, default is ``False``.                      |
+----------------+-------------------------------------------------------+
| norm_state     | Whether to use BatchNorm for normalization to prevent |
|                | gradient explosion, bool type, default is ``False``.  |
+----------------+-------------------------------------------------------+
| mode           | Specific contents are the same as :ref:`lif-lifplus`  |
|                | chapter's Table Initialization/Adjustable Parameters  |
| memb_mode      | of Lif.                                               |
|                |                                                       |
| soma_params    |                                                       |
|                |                                                       |
| noise          |                                                       |
|                |                                                       |
| input_accum    |                                                       |
|                |                                                       |
| rev_volt       |                                                       |
|                |                                                       |
| fire_refrac    |                                                       |
|                |                                                       |
| spike_init     |                                                       |
|                |                                                       |
| trig_current   |                                                       |
|                |                                                       |
| memb_decay     |                                                       |
+----------------+-------------------------------------------------------+
| spike_func     | [Data Type] callable                                  |
|                |                                                       |
|                | Gradient surrogate function used by neurons           |
+----------------+-------------------------------------------------------+
| use_inner_loop | [Data Type] bool type                                 |
|                |                                                       |
|                | Whether to use neuron inner loop mode                 |
+----------------+-------------------------------------------------------+
| it_batch       | [Data Type] int type                                  |
|                |                                                       |
|                | Inner loop mode's ``batch_size``, use ``1`` during    |
|                | APU inference, can be defined freely during training  |
+----------------+-------------------------------------------------------+


Lif1d/LifPlus1d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to FcLif/FcLifPlus without projection, meaning there is no nn.Linear layer, so input_channel, hidden_channel, and feed_back parameters are not needed.

Contains Convolutional Synapse 2D Data Processing Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conv2dLif
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherits from ``torch.nn.Module``. In the ``init`` method, initial values are assigned to the parameters involved in Lif, and the ``forward`` method is completely consistent with FcLif. It can only process a single time step input during use.

The initialization/adjustable parameters for the ConvLif2d layer are listed in the table below.

Table: Conv2dLif Layer Structure Parameter Description

+-----------------+-------------------------------------------------------+
| Parameter       | Meaning                                               |
+=================+=======================================================+
| input_channel   | Number of input layer channels, int type.             |
+-----------------+-------------------------------------------------------+
| hidden_channel  | Number of hidden or output layer channels, int.       |
+-----------------+-------------------------------------------------------+
| kernel_size     | Size of convolution kernel, int type.                 |
+-----------------+-------------------------------------------------------+
| stride          | Convolution stride, int type, default is ``1``.       |
+-----------------+-------------------------------------------------------+
| padding         | Convolution padding, int type, default is ``0``.      |
+-----------------+-------------------------------------------------------+
| dilation        | Dilation factor of convolution kernel, the spacing    |
|                 | between convolution kernel elements, int type,        |
|                 | default is ``1``.                                     |
+-----------------+-------------------------------------------------------+
| groups          | Number of blocked connections from input channels     |
|                 | to output channels, int type, default is ``1``.       |
+-----------------+-------------------------------------------------------+
| feed_back       | Whether to use feedback loops linking hidden layers   |
|                 | bool type, default is ``False``.                      |
+-----------------+-------------------------------------------------------+
| norm_state      | Whether to use BatchNorm for normalization to         |
|                 | prevent gradient explosion, bool type, default is     |
|                 | ``True``.                                             |
+-----------------+-------------------------------------------------------+
| mode            | Specific contents are the same as :ref:`lif-lifplus`  |
|                 | chapter's Table Initialization/Adjustable Parameters  |
| memb_mode       | of Lif.                                               |
|                 |                                                       |
| soma_params     |                                                       |
|                 |                                                       |
| noise           |                                                       |
+-----------------+-------------------------------------------------------+
| spike_func      | [Data Type] callable                                  |
|                 |                                                       |
|                 | Gradient surrogate function used by neurons           |
+-----------------+-------------------------------------------------------+
| use_inner_loop  | [Data Type] bool type                                 |
|                 |                                                       |
|                 | Whether to use neuron inner loop mode                 |
+-----------------+-------------------------------------------------------+
| it_batch        | [Data Type] int type                                  |
|                 |                                                       |
|                 | Inner loop mode's ``batch_size``, use ``1`` during    |
|                 | APU inference, can be defined freely during training. |
+-----------------+-------------------------------------------------------+

Conv2dLifPlus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherits from ``torch.nn.Module``. In the ``init`` method, initial values are assigned to the parameters involved in LifPlus, and the ``forward`` method is completely consistent with FcLifPlus. It can only process a single time step input during use.

The initialization/adjustable parameters for the ConvLifPlus2d layer are listed in the table below.

Table: Conv2dLif Layer Structure Parameter Description

+-----------------+-------------------------------------------------------+
| Parameter       | Meaning                                               |
+=================+=======================================================+
| input_channel   | Number of input layer channels, int type.             |
+-----------------+-------------------------------------------------------+
| hidden_channel  | Number of hidden or output layer channels, int .      |
+-----------------+-------------------------------------------------------+
| kernel_size     | Size of convolution kernel, int type.                 |
+-----------------+-------------------------------------------------------+
| stride          | Convolution stride, int type, default is ``1``.       |
+-----------------+-------------------------------------------------------+
| padding         | Convolution padding, int type, default is ``0``.      |
+-----------------+-------------------------------------------------------+
| dilation        | Dilation factor of convolution kernel, the spacing    |
|                 | between convolution kernel elements, int type,        |
|                 | default is ``1``.                                     |
+-----------------+-------------------------------------------------------+
| groups          | Number of blocked connections from input channels     |
|                 | to output channels, int type, default is ``1``.       |
+-----------------+-------------------------------------------------------+
| feed_back       | Whether to use feedback loops linking hidden layers   |
|                 | bool type, default is ``False``.                      |
+-----------------+-------------------------------------------------------+
| norm_state      | Whether to use BatchNorm for normalization to         |
|                 | prevent gradient explosion, bool type, default is     |
|                 | ``True``.                                             |
+-----------------+-------------------------------------------------------+
| mode            | Specific contents are the same as :ref:`lif-lifplus`  |
|                 | chapter's Table Initialization/Adjustable Parameters  |
| memb_mode       | of Lif.                                               |
|                 |                                                       |
| soma_params     |                                                       |
|                 |                                                       |
| noise           |                                                       |
|                 |                                                       |
| input_accum     |                                                       |
|                 |                                                       |
| rev_volt        |                                                       |
|                 |                                                       |
| fire_refrac     |                                                       |
|                 |                                                       |
| spike_init      |                                                       |
|                 |                                                       |
| trig_current    |                                                       |
|                 |                                                       |
| memb_decay      |                                                       |
+-----------------+-------------------------------------------------------+
| spike_func      | [Data Type] callable                                  |
|                 |                                                       |
|                 | Gradient surrogate function used by neurons           |
+-----------------+-------------------------------------------------------+
| use_inner_loop  | [Data Type] bool type                                 |
|                 |                                                       |
|                 | Whether to use neuron inner loop mode                 |
+-----------------+-------------------------------------------------------+
| it_batch        | [Data Type] int type                                  |
|                 |                                                       |
|                 | Inner loop mode's ``batch_size``, use ``1`` during    |
|                 | APU inference, can be defined freely during training. |
+-----------------+-------------------------------------------------------+

Lif2d/LifPlus2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to Conv2dLif/Conv2dLifPlus without projection, meaning there is no 
nn.Conv2d layer, so convolution-related parameters like input_channel, 
hidden_channel, kernel_size, etc., are not needed.

