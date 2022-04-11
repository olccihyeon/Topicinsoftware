
for TRAIN in "l2" "ewc"
do
for LAMB in 0.001 0.01 0.1 1 10 40 100 400 700 1000
do
    	python3 main.py --trainer $TRAIN --dataset MNIST --nepochs 20 --lr 0.001 --lamb $LAMB
done
done
