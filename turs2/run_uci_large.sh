#!/bin/bash

data_names2=("anuran" "avila" "magic" "waveform")

# Define array of fold numbers
fold_numbers=(0 1 2 3 4)

# Run the script for each data name in the second list, and for each fold number
for data_name in "${data_names2[@]}"; do
    for fold_number in "${fold_numbers[@]}"; do
        python3 exp_predictive_perf.py "$data_name" "$fold_number" &
    done
done

# Wait for all background processes to finish
wait

