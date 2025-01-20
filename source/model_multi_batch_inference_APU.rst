Batch Inference with Multiple Batches (to be executed on Lynxi brain-inspired systems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention:: For batch size > 1, use V1 compiler. The default batch size is 1, and supported batch sizes are 2, 4, 7, 14, 28. Smaller models can support up to 28batch. larger models may report the error 'Models with unequal numbers of batch and n_slice are not supported yet. Please reduce the batchsize number!' due to the compiler's underlying cutoff. please reduce the batchsize value!

+--------------+----------------------------------------------------------+
| 应用         | 执行脚本(以2batch为例)                                   |
+==============+==========================================================+
| DVS-MNIST    | python3 test.py \-\-config                               |
|              | clif3fc3dm_itout-b16x1-dvsmnist \-\-checkpoint           |
|              | best_clif3.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
|              |                                                          |
|              | python3 test.py \-\-config                               |
|              | clif5fc2dm_itout-b16x1-dvsmnist \-\-checkpoint           |
|              | best_clif5.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| DVS-Gesture  | python3 test.py \-\-config                               |
|              | clif3flif2dg_itout-b16x1-dvsgesture \-\-checkpoint       |
|              | best_clif3.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
|              |                                                          |
|              | python3 test.py \-\-config                               |
|              | clif7fc1dg_itout-b16x1-dvsgesture \-\-checkpoint         |
|              | best_clif7.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| CIFAR10-DVS  | python3 test.py \-\-config                               |
|              | clif5fc2cd_itout-b64x1-cifar10dvs \-\-checkpoint         |
|              | best_clif5.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
|              |                                                          |
|              | python3 test.py \-\-config                               |
|              | clif7fc1cd_itout-b64x1-cifar10dvs \-\-checkpoint         |
|              | best_clif7.pth \-\-use_lyngor 1 \-\-use_legacy 0         |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| Jester       | python3 test.py \-\-config                               |
|              | resnetlif18-itout-b20x4-16-jester \-\-checkpoint         |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0             |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| Luna16Cls    | python3 test.py \-\-config                               |
|              | clif3fc3lc_itout-b16x1-luna16cls \-\-checkpoint          |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0             |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| RGB-Gesture  | python3 test.py \-\-config                               |
|              | clif3flif2rg_itout-b16x1-rgbgesture \-\-checkpoint       |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0             |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+
| ESImagenet   | python3 test.py \-\-config                               |
|              | resnetlif18-itout-b64x4-esimagene \-\-checkpoint         |
|              | latest.pth \-\-use_lyngor 1 \-\-use_legacy 0             |
|              | \-\-v 1 \-\-b 2                                          |
+--------------+----------------------------------------------------------+