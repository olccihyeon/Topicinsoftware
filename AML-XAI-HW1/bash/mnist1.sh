GPU=0

CUDA_VISIBLE_DEVICES=$GPU python3 main.py --trainer vanilla --dataset MNIST --nepochs 20 --lr 0.001 --seed 1
