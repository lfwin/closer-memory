
#for train_size in $(seq 20000 10000 100000)
for train_size in 60000
do
	for lr in 0.1
	do
		echo "Python script called lr=$lr train_size=$train_size"
		python random_shift_RSU.py 16 1 $lr $train_size
        done
done



