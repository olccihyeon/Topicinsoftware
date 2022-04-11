

for LAMB in 0.001 0.01 0.1 1 10 40 100 400 700 1000
do
       	python3 main.py --trainer ewc --dataset CIFAR100 --nepochs 60 --lr 0.001 --lamb $LAMB
done



