Infrared Weak Target Identification (To be executed on Lynxi brain-inspired systems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------+--------------------------------------------------------+
| Step         | Execution Script                                       |
+==============+========================================================+
| PCNN         | In the *applications/dvsdetection/pcnn* directory:     |
|              |                                                        |
|              | python3 pcnn_sim.py \-\-compile_apu 0 \-\-device apu:0 |
|              | \-\-render 1                                           |
+--------------+--------------------------------------------------------+
| pcnn_det     | In the *applications/dvsdetection/pcnn_det* directory: |
|              |                                                        |
|              | python3 pcnn_det_sim.py \-\-compile_apu 0 \-\-device   |
|              | apu:0 \-\-render 1                                     |
+--------------+--------------------------------------------------------+