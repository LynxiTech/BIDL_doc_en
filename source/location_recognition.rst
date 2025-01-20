Location Recognition (If performing APU inference, execute on Lynxi brain-inspired systems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------+-------------------------------------------------------+
| Function     | Execution Script                                      |
+==============+=======================================================+
| Model        | In the *applications/robotics/placerecog/src/main*    |
| compilation  | directory:                                            |
| only, no APU |                                                       |
| inference    | python3 main_inference.py \-\-config_file             |
|              | ../config/room_setting.ini \-\-use_lyngor 1           |
|              | \-\-use_legacy 0 --c 1                                |
+--------------+-------------------------------------------------------+
| Model        | In the *applications/robotics/placerecog/src/main*    |
| compilation  | directory:                                            |
| and APU      |                                                       |
| inference    | python3 main_inference.py \-\-config_file             |
|              | ../config/room_setting.ini \-\-use_lyngor 1           |
|              | \-\-use_legacy 0 \-\-c 0                              |
+--------------+-------------------------------------------------------+
| APU          | In the *applications/robotics/placerecog/src/main*    |
| inference    | directory:                                            |
| only         |                                                       |
|              | python3 main_inference.py \-\-config_file             |
|              | ../config/room_setting.ini \-\-use_lyngor 0           |
|              | \-\-use_legacy 1 \-\-c 0                              |
+--------------+-------------------------------------------------------+