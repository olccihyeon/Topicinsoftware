GPU=1

CUDA_VISIBLE_DEVICES=$GPU python3 main.py --trainer vanilla --dataset CIFAR100 --nepochs 60 --lr 0.001 --seed 2

