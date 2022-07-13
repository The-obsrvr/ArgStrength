#!/bin/bash

python training.py --task_name=STLAS_only_swanson --gpus 0 --sampling_strategy balanced --train_batch_size=64
python training.py --task_name=STLAS_LOO_swanson --gpus 0 --sampling_strategy balanced --train_batch_size=48
#
#python training.py --task_name=STLAS_only_toledo --gpus 3 --sampling_strategy balanced --train_batch_size=32
#python training.py --task_name=STLAS_only_toledo --use_topic --gpus 3 --sampling_strategy balanced --train_batch_size=32
#
#python training.py --task_name=STLAS_only_webis --gpus 3 --sampling_strategy balanced --train_batch_size=32
#python training.py --task_name=STLAS_only_webis --use_topic --gpus 3 --sampling_strategy balanced --train_batch_size=32
