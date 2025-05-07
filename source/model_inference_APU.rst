Model Inference (To be executed on Lynxi brain-inspired systems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------+-------------------------------------------------------------+
| 应用         | 执行脚本                                                    |
+==============+=============================================================+
| DVS-MNIST    | python3 test.py \-\-config                                  |
|              | clif3fc3dm_itout-b16x1-dvsmnist \-\-checkpoint              |
|              | best_clif3.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | clif5fc2dm_itout-b16x1-dvsmnist \-\-checkpoint              |
|              | best_clif5.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
+--------------+-------------------------------------------------------------+
| DVS-Gesture  | python3 test.py \-\-config                                  |
|              | clif3flif2dg_itout-b16x1-dvsgesture \-\-checkpoint          |
|              | best_clif3.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | clif7fc1dg_itout-b16x1-dvsgesture \-\-checkpoint            |
|              | best_clif7.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
+--------------+-------------------------------------------------------------+
| CIFAR10-DVS  | python3 test.py \-\-config                                  |
|              | clif5fc2cd_itout-b64x1-cifar10dvs \-\-checkpoint            |
|              | best_clif5.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | clif7fc1cd_itout-b64x1-cifar10dvs \-\-checkpoint            |
|              | best_clif7.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | resnetlif50-lite-itout-b8x1-cifar10dvs \-\-checkpoint       |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | resnetlif50-itout-b8x1-cifar10dvs \-\-checkpoint            |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | resnetlif50-it-b16x1-cifar10dvs \-\-checkpoint              |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
+--------------+-------------------------------------------------------------+
| IMDB         | python3 test.py \-\-config fasttext-b16x1-imdb              |
|              | \-\-checkpoint latest.pth \-\-use_lyngor 1 \-\-use_legacy 0 |
+--------------+-------------------------------------------------------------+
| Jester       | python3 test.py \-\-config                                  |
|              | resnetlif18-itout-b20x4-16-jester \-\-checkpoint            |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | resnetlif18-lite-itout-b20x4-16-jester \-\-checkpoint       |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
+--------------+-------------------------------------------------------------+
| Luna16Cls    | python3 test.py \-\-config                                  |
|              | clif3fc3lc_itout-b16x1-luna16cls \-\-checkpoint             |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
+--------------+-------------------------------------------------------------+
| RGB-Gesture  | python3 test.py \-\-config                                  |
|              | clif3flif2rg_itout-b16x1-rgbgesture \-\-checkpoint          |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
+--------------+-------------------------------------------------------------+
| ESImagenet   | python3 test.py \-\-config                                  |
|              | resnetlif18-itout-b64x4-esimagenet \-\-checkpoint           |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0                |
+--------------+-------------------------------------------------------------+
| Cann         | In the path *applications/videotracking/CANN/*:             |
|              |                                                             |
|              | python3 compile.py                                          |
|              |                                                             |
|              | python3 infer_apu.py                                        |
+--------------+-------------------------------------------------------------+
| Music        | In the path *applications/onlinelearning/hebb/              |
| Recognition  | music_recognization*                                        |
|              | python3 main.py \-\-use_lyngor 1 \-\-use_legacy 0           |
+--------------+-------------------------------------------------------------+
| ZO-SGD       | In the path *applications/onlinelearning/ZO-SGD/*:          |
|              |                                                             |
|              | python3 main.py \-\-use_lyngor 1 \-\-use_legacy 0           |
+--------------+-------------------------------------------------------------+
| MNIST        | python3 test.py \-\-config                                  |
| Encoding     | clif3fc3mnrate_itout-b128x1-mnist \-\-checkpoint            |
|              | latest_rate.pth \-\-use_lyngor 1 \-\-use_legacy 0           |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | clif3fc3mntime_itout-b128x1-mnist \-\-checkpoint            |
|              | latest_time.pth \-\-use_lyngor 1 \-\-use_legacy 0           |
|              |                                                             |
|              | python3 test.py \-\-config                                  |
|              | clif3fc3mnpop_itout-b64x1-mnist \-\-checkpoint              |
|              | latest_pop.pth \-\-use_lyngor 1 \-\-use_legacy 0            |
+--------------+-------------------------------------------------------------+
| Spike-driven | with compile: python3 lynxi_inference.py \-\-batch_size 1   |
| Transformer  |  \-\-model metaspikformer_8_512  \-\-time_steps 4           |
| v2           |  \-\-device apu:0  \-\-compile True                         |
|              | use compiled: python3 lynxi_inference.py \-\-batch_size 1   |
|              |  \-\-model metaspikformer_8_512  \-\-time_steps 4           |
|              |  \-\-device apu:0                                           |
|              | executed in applications\/classification\/image\/           |
|              | spikeformerv2\/classification\/lynchip                      |
+--------------+-------------------------------------------------------------+
