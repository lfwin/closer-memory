

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"



for lr in 0.02
do
	echo "Python script called lr=$lr"
	CUDA_VISIBLE_DEVICES=0,1  python CIFAR10_shift_RSU.py 16 1 $lr
done


#python CIFAR10_relu.py 16 1 0.1
