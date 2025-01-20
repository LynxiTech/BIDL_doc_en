Execution Script Summary for Factory Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution via GPU/CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------------------------------------------------------+
| Neuron Model            | Execution Script                                        |
+=========================+=========================================================+
| lif                     | python3 test.py \-\-neuron lif \-\-use_lyngor 0         |
|                         | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| adex                    | python3 test.py \-\-neuron adex \-\-use_lyngor 0        |
|                         | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| izhikevich              | python3 test.py \-\-neuron izhikevich \-\-use_lyngor 0  |
|                         | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| Multicompartment Neuron | python3 test.py \-\-neuron multicompartment             |
|                         | \-\-use_lyngor 0 \-\-use_legacy 0 \-\-use_gpu 1         |
|                         | \-\-plot 0                                              |
+-------------------------+---------------------------------------------------------+
| Hodgkin-Huxley          | python3 test.py \-\-neuron hh \-\-use_lyngor 0          |
|                         | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| Multi-cluster Model     | python3 test.py \-\-neuron multicluster                 |
|                         | \-\-use_lyngor 0 \-\-use_legacy 0 \-\-use_gpu 1         |
|                         | \-\-plot 0                                              |
+-------------------------+---------------------------------------------------------+

Execution via Lynxi brain-inspired systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+---------------------------------------------------------+
| Neuron Model            | Execution Script                                        |
+=========================+=========================================================+
| lif                     | python3 test.py \-\-neuron lif \-\-use_lyngor 1         |
|                         | \-\-use_legacy 0 \-\-use_gpu 0 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| adex                    | python3 test.py \-\-neuron adex \-\-use_lyngor 1        |
|                         | \-\-use_legacy 0 \-\-use_gpu 0 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| izhikevich              | python3 test.py \-\-neuron izhikevich \-\-use_lyngor 1  |
|                         | \-\-use_legacy 0 \-\-use_gpu 0 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| Multicompartment Neuron | python3 test.py \-\-neuron multicompartment             |
|                         | \-\-use_lyngor 1 \-\-use_legacy 0 \-\-use_gpu 0         |
|                         | \-\-plot 0                                              |
+-------------------------+---------------------------------------------------------+
| Hodgkin-Huxley          | python3 test.py \-\-neuron hh \-\-use_lyngor 1          |
|                         | \-\-use_legacy 0 \-\-use_gpu 0 \-\-plot 0               |
+-------------------------+---------------------------------------------------------+
| Multi-cluster Model     | python3 test.py \-\-neuron multicluster                 |
|                         | \-\-use_lyngor 1 \-\-use_legacy 0 \-\-use_gpu 0         |
|                         | \-\-plot 0                                              |
+-------------------------+---------------------------------------------------------+

Execution via Lynxi brain-inspired systems using Historical Compilation Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+------------------------------------------------------------+
| Neuron Model      | Execution Script                                           |
+===================+============================================================+
| lif               | python3 test.py \-\-neuron lif \-\-use_lyngor 1            |
|                   | \-\-use_legacy 1 \-\-use_gpu 0 \-\-plot 0                  |
+-------------------+------------------------------------------------------------+
| adex              | python3 test.py \-\-neuron adex \-\-use_lyngor 1           |
|                   | \-\-use_legacy 1 \-\-use_gpu 0 \-\-plot 0                  |
+-------------------+------------------------------------------------------------+
| izhikevich        | python3 test.py \-\-neuron izhikevich \-\-use_lyngor 1     |
|                   | \-\-use_legacy 1 \-\-use_gpu 0 \-\-plot 0                  |
+-------------------+------------------------------------------------------------+
| Multicompartment  | python3 test.py \-\-neuron multicompartment                |
| Neuron            | \-\-use_lyngor 1 \-\-use_legacy 1 \-\-use_gpu 0            |
|                   | \-\-plot 0                                                 |
+-------------------+------------------------------------------------------------+
| Hodgkin-Huxley    | python3 test.py \-\-neuron hh \-\-use_lyngor 1             |
|                   | \-\-use_legacy 1 \-\-use_gpu 0 \-\-plot 0                  |
+-------------------+------------------------------------------------------------+
| Multi-cluster     | python3 test.py \-\-neuron multicluster \-\-use_lyngor 1   |
| Model             | \-\-use_legacy 1 \-\-use_gpu 0 \-\-plot 0                  |
+-------------------+------------------------------------------------------------+

Execution via GPU/CPU and Lynxi brain-inspired systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+---------------------------------------------------------------+
| Neuron Model      | Execution Script                                              |
+===================+===============================================================+
| lif               | python3 test.py \-\-neuron lif \-\-use_lyngor 1               |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0                     |
+-------------------+---------------------------------------------------------------+
| adex              | python3 test.py \-\-neuron adex \-\-use_lyngor 1              |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0                     |
+-------------------+---------------------------------------------------------------+
| izhikevich        | python3 test.py \-\-neuron izhikevich \-\-use_lyngor 1        |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0                     |
+-------------------+---------------------------------------------------------------+
| Multicompartment  | python3 test.py \-\-neuron multicompartment \-\-use_lyngor    |
| Neuron            | 1 \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0                   |
+-------------------+---------------------------------------------------------------+
| Hodgkin-Huxley    | python3 test.py \-\-neuron hh \-\-use_lyngor 1 \-\-use_legacy |
|                   | 0 \-\-use_gpu 1 \-\-plot 0                                    |
+-------------------+---------------------------------------------------------------+
| Multi-cluster     | python3 test.py \-\-neuron multicluster \-\-use_lyngor 1      |
| Model             | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 0                     |
+-------------------+---------------------------------------------------------------+

Execution via GPU/CPU and Lynxi brain-inspired systems, with Spiking Statistics Curve Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+------------------------------------------------------------+
| Neuron Model      | Execution Script                                           |
+===================+============================================================+
| lif               | python3 test.py \-\-neuron lif \-\-use_lyngor 1            |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                  |
+-------------------+------------------------------------------------------------+
| adex              | python3 test.py \-\-neuron adex \-\-use_lyngor 1           |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                  |
+-------------------+------------------------------------------------------------+
| izhikevich        | python3 test.py \-\-neuron izhikevich \-\-use_lyngor 1     |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                  |
+-------------------+------------------------------------------------------------+
| Multicompartment  | python3 test.py \-\-neuron multicompartment \-\-use_lyngor |
| Neuron            | 1 \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                |
+-------------------+------------------------------------------------------------+
| Hodgkin-Huxley    | python3 test.py \-\-neuron hh \-\-use_lyngor 1             |
|                   | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                  |
+-------------------+------------------------------------------------------------+
| Multi-cluster     | python3 test.py \-\-neuron multicluster \-\-use_lyngor 1   |
| Model             | \-\-use_legacy 0 \-\-use_gpu 1 \-\-plot 1                  |
+-------------------+------------------------------------------------------------+

When using Multi-cluster Model + STDP, execute the test_stdp.py script with the following configuration:

+----------------+----------+--------------------------------------------------------+
| Device         | Plot     | Execution Script                                       |
+================+==========+========================================================+
| Lynxi          | Yes      | python3 test_stdp.py \-\-use_lyngor 1                  |
| Brain-Inspired |          | \-\-use_legacy 0 \-\-use_gpu 0 \-\-plot 1              |
+----------------+----------+--------------------------------------------------------+
| GPU/CPU        | Yes      | python3 test_stdp.py \-\-use_lyngor 0 \-\-use_legacy 0 |
|                |          | \-\-use_gpu 1 \-\-plot 1                               |
+----------------+----------+--------------------------------------------------------+
| Lynxi          | No       | python3 test_stdp.py \-\-use_lyngor 1 \-\-use_legacy 0 |
| Brain-Inspired |          | \-\-use_gpu 0 \-\-plot 0                               |
+----------------+----------+--------------------------------------------------------+
| GPU/CPU        | No       | python3 test_stdp.py \-\-use_lyngor 0 \-\-use_legacy 0 |
|                |          | \-\-use_gpu 1 \-\-plot 0                               |
+----------------+----------+--------------------------------------------------------+
