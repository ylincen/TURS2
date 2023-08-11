#!/bin/bash

data_names0=("Vehicle" "DryBeans" "glass" "pendigits" "HeartCleveland")

fold_numbers=(0 1 2 3 4)

for data_name in "${data_names0[@]}"; do
    for fold_number in "${fold_numbers[@]}"; do
        python3 run_uci.py "$data_name" "$fold_number" &
    done
done

wait

## Define array of data names for the first python script
#data_names1=("chess" "iris" "backnote" "contracept" "ionosphere" "car" "tic-tac-toe" "wine" "diabetes")
#
## Run the script for each data name in the first list
#for data_name in "${data_names1[@]}"; do
#    python3 run_uci.py "$data_name" &
#done
#
#wait
#
#data_names2=("anuran" "avila" "magic" "waveform")
#
#fold_numbers=(0 1 2 3 4)
#
## Run the script for each data name in the second list, and for each fold number
#for data_name in "${data_names2[@]}"; do
#    for fold_number in "${fold_numbers[@]}"; do
#        python3 run_uci.py "$data_name" "$fold_number" &
#    done
#done
#
## Wait for all background processes to finish
#wait

