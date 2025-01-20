Software Installation
===============================================================================

Acquiring Software
--------------------------------------------------------------------------------

Visit the `BIDL open-source page <https://github.com/openBII/BIDL>`__ 
to obtain the source code project.

**Installation Package**

Clone the https://github.com/openBII/BIDL repository to local directory, such as:
  
  git clone https://github.com/openBII/BIDL.git

Note that this package only include project source code; 
for code folder details, refer to :ref:`bidl_folder`.

**Resources Package**

The resource package includes compressed files, which can be downloaded 
from Baidu Net-disk, after download, please unzip them to the same 
directory as in the baidu net-disk.

The resource package includes the dataset, pre-trained weights 
(executed in GPU/CPU) and compiled
model files (executed in lynxi chip) for BIDL, including:

- data: datasets;
- model_files: model files; 
- weight_files: weight files. 

Deploy Training Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU training environment this framework depends on is mainly PyTorch, so PyTorch needs to be installed first.

First, install the GPU version of PyTorch. 
::

  pip install torch torchvision torchaudio 

Deploy Compilation Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Lyngor, which is used for compiling models, only supports the CPU version of PyTorch, you need to reconfigure the CPU version of the PyTorch environment.
Additionally, compilation and deployment require support from Lyngor and SDK software, as well as APU hardware.

Execute the following command to install the latest stable version of PyTorch provided by the PyTorch official website.

::

  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

Validation
--------------------------------------------------------------------------------

GPU Environment Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the ``pip list`` command to check whether the installed packages fully include those specified in *requirements.txt*.

Import the corresponding packages in the Python environment to verify correct installation.

::

  >>> torch.__version__
  '2.4.0+cu121'

CPU Environment Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the ``pip list`` command to check whether the installed packages fully include those specified in *requirements.txt*.

Import the corresponding packages in the Python environment to verify correct installation.

::

  >>> torch.__version__
  '2.4.0+cpu'

.. _bidl_folder:

Directory Description
--------------------------------------------------------------------------------

After BIDL is installed, refer to the table below for the directory structure and description.

Table: BIDL Directory Structure

+-----------------+-----------------------------------------------------------+
| Directory/File  | Description                                               |
+=================+===========================================================+
| applications    | The storage directory for application cases, which        |
|                 | includes the following categories:                        |
|                 |                                                           |
|                 | - classification (Classification)                         |
|                 | - dvsdetection (Detection)                                |
|                 | - neuralsim (Brain-like Neuron and SNN Simulation)        |
|                 | - onlinelearning (Online Learning)                        |
|                 | - videotracking (Visual Tracking)                         |
|                 | - robotics (Robotics)                                     |
|                 | - functionalBII (Functional Level Brain-Inspired          |
|                 |   Intelligence)                                           |
|                 |                                                           |
|                 | Among them, functionalBII and onlinelearning are only     |
|                 | provided in the commercial version.                       |
+-----------------+-----------------------------------------------------------+
| lynadapter      | Directory for storing Lynxi chip interface files          |
+-----------------+-----------------------------------------------------------+
| requirements    | Specifies third-party software packages required for the  |
|                 | project                                                   |
+-----------------+-----------------------------------------------------------+
| tools           | Includes training and inference scripts                   |
+-----------------+-----------------------------------------------------------+
| tutorial        | Path for storing demonstration cases                      |
+-----------------+-----------------------------------------------------------+
| utils           | Path for storing utility functions                        |
+-----------------+-----------------------------------------------------------+
| deploy          | Path for storing demos that operate without the framework |
+-----------------+-----------------------------------------------------------+