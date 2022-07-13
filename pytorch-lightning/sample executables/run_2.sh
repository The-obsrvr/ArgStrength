#!/bin/bash

python training.py --task_name=STLAS_only_gretz --gpus 0 --sampling_strategy in-topic
python training.py --task_name=STLAS_only_gretz --gpus 0 --sampling_strategy cross-topic
python training.py --task_name=STLAS_LOO_gretz --gpus 0 --sampling_strategy in-topic
python training.py --task_name=STLAS_LOO_gretz --gpus 0 --sampling_strategy cross-topic