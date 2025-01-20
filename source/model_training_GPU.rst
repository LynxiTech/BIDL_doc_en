Model Training (To be executed on GPU devices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------+-------------------------------------------------------+
| 应用         | 执行脚本                                              |
+==============+=======================================================+
| DVS-MNIST    | python3 train.py \-\-config                           |
|              | clif3fc3dm_itout-b16x1-dvsmnist                       |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | clif5fc2dm_itout-b16x1-dvsmnist                       |
+--------------+-------------------------------------------------------+
| DVS-Gesture  | python3 train.py \-\-config                           |
|              | clif3flif2dg_itout-b16x1-dvsgesture                   |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | clif7fc1dg_itout-b16x1-dvsgesture                     |
+--------------+-------------------------------------------------------+
| CIFAR10-DVS  | python3 train.py \-\-config                           |
|              | clif5fc2cd_itout-b64x1-cifar10dvs                     |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | clif7fc1cd_itout-b64x1-cifar10dvs                     |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | resnetlif50-lite-itout-b8x1-cifar10dvs                |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | resnetlif50-itout-b8x1-cifar10dvs                     |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | resnetlif50-it-b16x1-cifar10dvs                       |
+--------------+-------------------------------------------------------+
| IMDB         | python3 train.py \-\-config fasttext-b16x1-imdb       |
+--------------+-------------------------------------------------------+
| Jester       | python3 train.py \-\-config                           |
|              | resnetlif18-itout-b20x4-16-jester                     |
+--------------+-------------------------------------------------------+
|              | python3 train.py \-\-config                           |
|              | resnetlif18-lite-itout-b20x4-16-jester                |
+--------------+-------------------------------------------------------+
| Luna16Cls    | python3 train.py \-\-config                           |
|              | clif3fc3lc_itout-b16x1-luna16cls                      |
+--------------+-------------------------------------------------------+
| RGB-Gesture  | python3 train.py \-\-config                           |
|              | clif3flif2rg_itout-b16x1-rgbgesture                   |
+--------------+-------------------------------------------------------+
| ESImagenet   | python3 train.py \-\-config                           |
|              | resnetlif18-itout-b64x4-esimagenet                    |
+--------------+-------------------------------------------------------+
| MNIST Coding | python3 train.py \-\-config                           |
|              | clif3fc3mnrate_itout-b128x1-mnist                     |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | clif3fc3mntime_itout-b128x1-mnist                     |
|              |                                                       |
|              | python3 train.py \-\-config                           |
|              | clif3fc3mnpop_itout-b64x1-mnist                       |
+--------------+-------------------------------------------------------+
| St-yolo      | In the path *applications/dvsdetection/st-yolo/*:     |
|              |                                                       |
|              | python3 train.py                                      |
+--------------+-------------------------------------------------------+
| High-speed-  | In the path *applications/dvsdetection/high-speed-    |
|              | turntable/*:                                          |
| turntable    |                                                       |
|              | python3 train.py                                      |
+--------------+-------------------------------------------------------+