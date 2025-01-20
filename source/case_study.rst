Case Study
========================================================================================

Classification
----------------------------------------------------------------------------------------

.. include:: classification_case.rst

Brain-inspired Tracking (Continuous Attractor) 
----------------------------------------------------------------------------------------

.. include:: brain_inspired_tracking_case_study.rst

DVS High-speed Target Detection
------------------------------------------------------------------------------------------

.. _st-yolo:

ST-YOLO Pedestrian Vehicle Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: ST-YOLO_pedestrian_vehicle_detection.rst

DVS High-Speed Turntable Object Detection with ST-Yolo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: dvs_high-speed_turntable_object_detection.rst

Location Recognition
------------------------------------------------------------------------------------------------

.. include:: location_recognition_case.rst

Small Target Segmentation and Detection with PCNN
------------------------------------------------------------------------------------------

.. include:: infrared_dim_target_identification.rst

Frequency/Time/Population Coding 
-----------------------------------------------------------------------------------

.. include:: frequency_time_population_code_support_cases.rst

Inner Loop SNN 
-----------------------------------------------------------------------------------

.. include:: internal_loop_support_case.rst


Summary of Used Networks in Above Cases
------------------------------------------------------------------------

.. _wxhmsdmx:

Models in Outer Loop Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Sequential

+----------------------+------------------------------------------------------+
| Model                | Description                                          |
+======================+======================================================+
| SeqClif3Fc3DmItout   | Sequential model for DVS-MNIST dataset, includes     |
|                      | ConvLif x3, FC x3;                                   |
+----------------------+------------------------------------------------------+
| SeqClif5Fc2DmItout   | Sequential model for DVS-MNIST dataset, includes     |
|                      | ConvLif x5, FC x2;                                   |
+----------------------+------------------------------------------------------+
| SeqClif3Flif2DgItout | Sequential model for DVS-Gesture dataset, includes   |
|                      | ConvLif x3, FcLif x2;                                |
+----------------------+------------------------------------------------------+
| SeqClif7Fc1DgItout   | Sequential model for DVS-Gesture dataset, includes   |
|                      | ConvLif x7, FC x1;                                   |
+----------------------+------------------------------------------------------+
| SeqClif5Fc2CdItout   | Sequential model for CIFAR10-DVS dataset, includes   |
|                      | ConvLif x5, FC x2;                                   |
+----------------------+------------------------------------------------------+
| SeqClif7Fc1CdItout   | Sequential model for CIFAR10-DVS dataset, includes   |
|                      | ConvLif x7, FC x1;                                   |
+----------------------+------------------------------------------------------+
| FastText             | For IMDB dataset;                                    |
+----------------------+------------------------------------------------------+
| SeqClif3Fc3LcItout   | Sequential model for Luna16Cls dataset, includes     |
|                      | ConvLif x3, FC x3;                                   |
+----------------------+------------------------------------------------------+

ResNet Series

+-------------------+--------------------------------------------------+
| Model             | Description                                      |
+===================+==================================================+
| ResNetLifItout-18 | Dataset agnostic, ReLU replaced by Lif, includes |
|                   | four stages, each with 2 residual blocks;        |
+-------------------+--------------------------------------------------+
| ResNetLifItout-50 | Dataset agnostic, ReLU replaced by Lif, includes |
|                   | four stages with residual blocks 3, 4, 6, and 3; |
+-------------------+--------------------------------------------------+

Models in Inner Loop Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Sequential

+-----------------+----------------------------------------------------------+
| Model           | Description                                              |
+=================+==========================================================+
| SeqClif5Fc2DmIt | Sequential model for DVS-MNIST dataset, includes ConvLif |
|                 | x5, FC x2;                                               |
+-----------------+----------------------------------------------------------+
| SeqClif7Fc1DgIt | Sequential model for DVS-Gesture dataset, includes       |
|                 | ConvLif x7, FC x1;                                       |
+-----------------+----------------------------------------------------------+
| SeqClif5Fc2CdIt | Sequential model for CIFAR10-DVS dataset, includes       |
|                 | ConvLif x5, FC x2;                                       |
+-----------------+----------------------------------------------------------+
| SeqClif7Fc1CdIt | Sequential model for CIFAR10-DVS dataset, includes       |
|                 | ConvLif x7, FC x1;                                       |
+-----------------+----------------------------------------------------------+
