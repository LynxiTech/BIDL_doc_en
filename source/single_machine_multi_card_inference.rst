Single Machine Multi Card Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-------------------------------------------------------+
| Application   | Execution Script                                      |
+===============+=======================================================+
| No Model      | python3 apuinfer_mutidevice.py \-\-config             |
| Partitioning  | resnetlif50-itout-b8x1-cifar10dvs                     |
+---------------+-------------------------------------------------------+
| Spatiotemporal| python3 apuinfer_mutidevice.py \-\-config             |
| Partitioning  | resnetlif50-itout-b8x1-cifar10dvs_mp                  |
+---------------+-------------------------------------------------------+