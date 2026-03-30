#!/bin/bash

# Define array of data names for the first python script

data_names1=("3_backdoor.npz")
folds=(0 1 2 3 4)
# Run the script for each data name in the first list
for data_name in "${data_names1[@]}"; do
    for fold in "${folds[@]}"; do     
        python run_ad_bench.py "$data_name" "$fold" &
    done
done

wait
