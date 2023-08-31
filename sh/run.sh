#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -o /work/gu14/k36093/energy_norm/result_cifar100_10tasks_2.txt
#PJM -g gk36
#PJM -j
#PJM --fs /work
#PJM -m b,s
#PJM --mail-list jinyas.wisteria@gmail.com
#PJM -N demo

module load cuda/11.3
# wandb login f1a9a859505bece805152bcd9bcb82def6b657b8


# /work/gu14/k36093/.pyenv/versions/3.8.0/envs/hello_world/bin/python /work/gu14/k36093/energy_norm/src/main.py --train_epochs 70 --retrain_epochs 50 --num_iters 3 --num_tasks 2 --num_classes 100 --num_classes_per_task 50 --dataset_name cifar100 --test_batch_size 5  --task_select_method free_energy
# /work/gu14/k36093/.pyenv/versions/3.8.0/envs/hello_world/bin/python /work/gu14/k36093/energy_norm/src/main.py --train_epochs 70 --retrain_epochs 50 --num_iters 3 --num_tasks 5 --num_classes 100 --num_classes_per_task 20 --dataset_name cifar100 --test_batch_size 5  --task_select_method free_energy
# /work/gu14/k36093/.pyenv/versions/3.8.0/envs/hello_world/bin/python /work/gu14/k36093/energy_norm/src/main.py --train_epochs 70 --retrain_epochs 50 --num_iters 3 --num_tasks 10 --num_classes 100 --num_classes_per_task 10 --dataset_name cifar100 --test_batch_size 5  --task_select_method free_energy
/work/gu14/k36093/.pyenv/versions/3.8.0/envs/hello_world/bin/python /work/gu14/k36093/energy_norm/src/main.py --train_epochs 70 --retrain_epochs 50 --num_iters 3 --num_tasks 10 --num_classes 100 --num_classes_per_task 10 --dataset_name cifar100 --test_batch_size 5  --task_select_method free_energy --alpha_conv 0.9 --model_name resnet18



