Data and Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This dataset was collected using Lynxi proprietary high-speed moving target data acquisition device, as depicted in the following figure. After collecting the DVS event stream data, it undergoes processes such as event stream framing, label generation, and dataset integration to form a complete DVS high-speed rotary target detection dataset. The dataset contains three types of targets: {0: 'Cross', 1: 'Triangle', 2: 'Circle'}.

.. figure:: _images/高速移动目标数据采集装置.png
   :alt: High-speed moving target data acquisition device

   Figure: High-speed moving target data acquisition device

The following figure demonstrates a visualization of a detection result.


.. figure:: _images/高速转盘检测可视化.jpeg
   :alt: High-speed rotary detection visualization

   Figure: High-speed rotary detection visualization

Network Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dvs_high-speed_rotary_target_detection utilizes the ST-YOLO network, based on spatiotemporal dynamics, for event stream data target detection. Unlike :ref:`st-yolo` for vehicle and pedestrian detection, the network head part in this task adopts the yolov5 head due to differences in dataset annotation formats.