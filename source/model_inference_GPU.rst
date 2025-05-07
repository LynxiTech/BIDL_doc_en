Model Inference (To be executed on GPU devices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------+-------------------------------------------------------------------------+
| 应用             | 执行脚本                                                            |
+==============+=========================================================================+
| DVS-MNIST        | python3 test.py \-\-config                                          |
|                  | clif3fc3dm_itout-b16x1-dvsmnist \-\-checkpoint                      |
|                  | best_clif3.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | clif5fc2dm_itout-b16x1-dvsmnist \-\-checkpoint                      |
|                  | best_clif5.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
+--------------+-------------------------------------------------------------------------+
| DVS-Gesture      | python3 test.py \-\-config                                          |
|                  | clif3flif2dg_itout-b16x1-dvsgesture \-\-checkpoint                  |
|                  | best_clif3.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | clif7fc1dg_itout-b16x1-dvsgesture \-\-checkpoint                    |
|                  | best_clif7.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
+--------------+-------------------------------------------------------------------------+
| CIFAR10-DVS      | python3 test.py \-\-config                                          |
|                  | clif5fc2cd_itout-b64x1-cifar10dvs \-\-checkpoint                    |
|                  | best_clif5.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | clif7fc1cd_itout-b64x1-cifar10dvs \-\-checkpoint                    |
|                  | best_clif7.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | resnetlif50-lite-itout-b8x1-cifar10dvs \-\-checkpoint               |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | resnetlif50-itout-b8x1-cifar10dvs \-\-checkpoint                    |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | resnetlif50-it-b16x1-cifar10dvs \-\-checkpoint                      |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
+------------------+---------------------------------------------------------------------+
| IMDB             | python3 test.py \-\-config fasttext-b16x1-imdb                      |
|                  | \-\-checkpoint latest.pth \-\-use_lyngor 0 \-\-use_legacy 0         |
+------------------+---------------------------------------------------------------------+
| Jester           | python3 test.py \-\-config                                          |
|                  | resnetlif18-itout-b20x4-16-jester \-\-checkpoint                    |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | resnetlif18-lite-itout-b20x4-16-jester \-\-checkpoint               |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
+------------------+---------------------------------------------------------------------+
| Luna16Cls        | python3 test.py \-\-config                                          |
|                  | clif3fc3lc_itout-b16x1-luna16cls \-\-checkpoint                     |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
+------------------+---------------------------------------------------------------------+
| RGB-Gesture      | python3 test.py \-\-config                                          |
|                  | clif3flif2rg_itout-b16x1-rgbgesture \-\-checkpoint                  |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
+------------------+---------------------------------------------------------------------+
| ESImagenet       | python3 test.py \-\-config                                          |
|                  | resnetlif18-itout-b64x4-esimagenet \-\-checkpoint                   |
|                  | latest.pth \-\-use_lyngor 0 \-\-use_legacy 0                        |
+------------------+---------------------------------------------------------------------+
| CANN             | In the path *applications/videotracking/CANN/*:                     |
|                  |                                                                     |
|                  | python3 infer_gpu.py                                                |
+------------------+---------------------------------------------------------------------+
| Music Recognition| In the path *applications/onlinelearning/hebb/music_recognization*  |
|                  |                                                                     |
|                  | python3 main.py \-\-use_lyngor 0 \-\-use_legacy 0                   |
+------------------+---------------------------------------------------------------------+
| Location         | In the path *applications/robotics/placerecog/src/main*:            |
| Recognition      |                                                                     |
|                  | python3 main_inference.py \-\-config_file                           |
|                  | ../config/room_setting.ini \-\-use_lyngor 0                         |
|                  | \-\-use_legacy 0 \-\-c 0                                            |
+------------------+---------------------------------------------------------------------+
| PCNN             | python3 pcnn_sim.py \-\-compile_apu 0 \-\-device cuda:0             |
|                  | \-\-render 1                                                        |
+------------------+---------------------------------------------------------------------+
| ZO-SGD           | In the path *applications/onlinelearning/ZO-SGD/*:                  |
|                  |                                                                     |
|                  | python3 main.py \-\-use_lyngor 0 \-\-use_legacy 0                   |
+------------------+---------------------------------------------------------------------+
| MNIST Encoding   | python3 test.py \-\-config                                          |
|                  | clif3fc3mnrate_itout-b128x1-mnist \-\-checkpoint                    |
|                  | latest_rate.pth \-\-use_lyngor 0 \-\-use_legacy 0                   |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | clif3fc3mntime_itout-b128x1-mnist \-\-checkpoint                    |
|                  | latest_time.pth \-\-use_lyngor 0 \-\-use_legacy 0                   |
|                  |                                                                     |
|                  | python3 test.py \-\-config                                          |
|                  | clif3fc3mnpop_itout-b64x1-mnist \-\-checkpoint                      |
|                  | latest_pop.pth \-\-use_lyngor 0 \-\-use_legacy 0                    |
+------------------+---------------------------------------------------------------------+
| St-yolo          | In the path *applications/dvsdetection/st-yolo/*:                   |
|                  |                                                                     |
|                  | python3 gpu_infer.py                                                |
|                  |                                                                     |
|                  | python3 gpu_show.py                                                 |
+------------------+---------------------------------------------------------------------+
| High-speed-      | In the path *applications/dvsdetection/high-speed-turntable/*       |
|                  |                                                                     |
| Turntable        |                                                                     |
|                  | python3 val.py                                                      |
|                  |                                                                     |
|                  | python3 detect.py                                                   |
+------------------+---------------------------------------------------------------------+
