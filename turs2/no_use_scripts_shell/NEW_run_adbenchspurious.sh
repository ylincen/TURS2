#!/bin/bash

# Define arrays for validity_check_ and spurious_type
validity_checks=("either" "none")
spurious_types=("dep")

# Define array of data names for the first python script
data_names1=("26_optdigits.npz" "34_smtp.npz" "28_pendigits.npz" "43_WDBC.npz" "36_speech.npz" "31_satimage-2.npz" "3_backdoor.npz" "38_thyroid.npz" "41_Waveform.npz" "23_mammography.npz" "40_vowels.npz" "25_musk.npz" "1_ALOI.npz" "14_glass.npz")

# Iterate over all combinations of validity_check_ and spurious_type
for validity_check_ in "${validity_checks[@]}"; do
    for spurious_type in "${spurious_types[@]}"; do
        for data_name in "${data_names1[@]}"; do
            echo "Running: $data_name with validity_check_=$validity_check_ and spurious_type=$spurious_type"
            python run_adbench_spurious.py "$data_name" "$validity_check_" "$spurious_type" &
        done
    done
done

# Wait for all background processes to finish
wait
