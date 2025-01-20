Quick Start
========================================================================================

This chapter uses dvs-mnist as an example to illustrate the usage of the BIDL project.

.. _choose_config_file:

Choose Configuration File
--------------------------------------------------------------------------------------------------------------

The dvs-mnist dataset uses the *clif3fc3dm_itout-b16x1-dvsmnist.py* configuration file. Since dvs-mnist 
belongs to the classification of dvs data, create a *dvs_mnist* folder in the *applications/classification/dvs* 
directory. Based on its model structure, create a *clif3fc3dm* folder under the new *dvs_mnist* folder, 
and finally place the configuration file in this path.

.. attention:: Configuration files are Python source code files, and their filenames have a .py suffix.

Prepare Dataset
--------------------------------------------------------------------------------------------------------

Check the dataset path set in the configuration file chosen in :ref:`choose_config_file`, and then create 
the corresponding dataset directory in the project's root directory.

::

   data = dict(
      samples_per_gpu=16,
      workers_per_gpu=2,
      train=dict(
         pipeline=train_pipeline
      ),
      val=dict(
         pipeline=test_pipeline,
         test_mode=True
      ),
      test=dict(
         pipeline=test_pipeline,
         test_mode=True
      )
   )

.. _mxxl:

Model Training
--------------------------------------------------------------------------------

In the *tools* directory, execute the following command to start model training.

::
   
   python train.py --config clif3fc3dm_itout-b16x1-dvsmnist
   
.. attention:: 
   
   The parameter ``--config`` followed by ``clif3fc3dm_itout-b16x1-dvsmnist`` is the name of the configuration file selected 
   in the :ref:`choose_config_file` section, without the .py.

After training starts, the configuration information involved in this training will be printed in sequence:

- Configuration information from the configuration file
- The path information for saving checkpoints.

      By default, checkpoint files are saved in the project root directory */work_dirs/clif3fc3dm_itout-b16x1-dvsmnist/*, 
      where a *latest.pth* file is generated for the last saved checkpoint of the epoch, and a *best.pth* file is created 
      to save the checkpoint with the best accuracy on the validation set. The directory tree is as follows:

      ::

         clif3fc3dm_itout-b16x1-dvsmnist/
         ├── 20240829_171552.log #All logs in this training process, such as loss, accuracy
         ├── best.pth #Checkpoint of the model with the highest accuracy on the validation set
         └── latest.pth #Checkpoint of the model from the last training round

- The following contents are model training log information. Below is the log information output in the first two epochs:

::

   2024-08-29 17:33:37,021 - cls - INFO - Epoch [1][100/452] lr:
   2.060e-02 top-1: 22.7500 top-5: 62.5000 loss: 2.1440

   2024-08-29 17:33:48,696 - cls - INFO - Epoch [1][200/452] lr:
   4.040e-02 top-1: 73.8750 top-5: 96.8125 loss: 1.1264

   2024-08-29 17:34:00,255 - cls - INFO - Epoch [1][300/452] lr:
   6.020e-02 top-1: 89.1875 top-5: 98.6250 loss: 0.8457

   2024-08-29 17:34:11,309 - cls - INFO - Epoch [1][400/452] lr:
   8.000e-02 top-1: 90.6250 top-5: 99.1250 loss: 0.7871

   2024-08-29 17:34:16,431 - cls - INFO - Saving checkpoint at 1 epochs

   2024-08-29 17:34:21,611 - cls - INFO - Epoch(val) [1][20] top-1:
   87.1275 top-5: 98.9238

   2024-08-29 17:34:31,200 - cls - INFO - Epoch [2][100/452] lr:
   9.980e-02 top-1: 94.3750 top-5: 99.6250 loss: 0.7186

   2024-08-29 17:34:41,470 - cls - INFO - Epoch [2][200/452] lr:
   9.980e-02 top-1: 94.6875 top-5: 99.6250 loss: 0.6894

   2024-08-29 17:34:51,961 - cls - INFO - Epoch [2][300/452] lr:
   9.980e-02 top-1: 97.3125 top-5: 99.8750 loss: 0.6128

   2024-08-29 17:35:02,393 - cls - INFO - Epoch [2][400/452] lr:
   9.980e-02 top-1: 97.3750 top-5: 99.9375 loss: 0.5926

   2024-08-29 17:35:07,965 - cls - INFO - Saving checkpoint at 2 epochs

   2024-08-29 17:35:14,549 - cls - INFO - Epoch(val) [2][20] top-1:
   98.1374 top-5: 100.0000

After the training, copy the best weight file to the weight file path in the resource package: 
*/weight_files/classification/clif3fc3dm/lif*

Model Inference (GPU)
--------------------------------------------------------------------------------

In the *tools* directory, run the following command for model inference.

::

   python test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint latest.pth --use_lyngor 0 --use_legacy 0

The inference process is as follows:

::

   100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2409/2409 [01:23<00:00, 28.85it/s]
   acc top-1: 99.21 top-5: 100.00
   gpu test speed = 576.6303 fps
   gpu test speed (without pipeline) = 583.3422 fps

Since the checkpoint files' save path is fixed by default during training, and the subdirectory is the same as the configuration file name, the parameter for ``--checkpoint`` only needs to specify the *pth* filename.

Model Inference (Lynxi brain-inspired systems)
--------------------------------------------------------------------------------

In the BIDL framework, a switch is used to toggle the backend execution environment. Specifically, when using *test.py*, configure as follows:

+----------------+------------------------------------------------------------+
| Configuration  | Description                                                |
+================+============================================================+
| \-\-use_lyngor | Used to specify whether to use Lyngor for compilation.     |
|                |                                                            |
|                | [Data Type] ENUM                                           |
|                |                                                            |
|                | [Value Range]                                              |
|                |                                                            |
|                | - 1: Use Lyngor for compilation, executed with Lynxi       |
|                |   brain-inspired systems;                                  |
|                | - 0: Do not use Lyngor for compilation (GPU execution).    |
+----------------+------------------------------------------------------------+
| \-\-use_legacy | Used to specify whether to directly load historical        |
|                | compilation artifacts, skipping the compilation step.      |
|                |                                                            |
|                | [Data Type] ENUM                                           |
|                |                                                            |
|                | [Value Range]                                              |
|                |                                                            |
|                | - 1: Load historical compilation artifacts, skip           |
|                |   compilation step;                                        |
|                | - 0: Do not load historical compilation artifacts, do      |
|                |   not skip compilation step.                               |
+----------------+------------------------------------------------------------+

Like this:

::

   python test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint latest.pth --use_lyngor 1 --use_legacy 0

.. attention:: You need to copy the trained files (located in *work_dirs*) and the validation dataset to Lynxi brain-inspiredcomputing devices before executing this script.