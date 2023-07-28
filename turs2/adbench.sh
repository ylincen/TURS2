#!/bin/bash

# Define array of data names for the first python script

data_names1=("26_optdigits.npz" "42_WBC.npz" "21_Lymphography.npz" "34_smtp.npz" "28_pendigits.npz" "43_WDBC.npz" "36_speech.npz" "31_satimage-2.npz" "3_backdoor.npz" "38_thyroid.npz" "41_Waveform.npz" "23_mammography.npz" "40_vowels.npz" "25_musk.npz" "1_ALOI.npz" "14_glass.npz")

# Run the script for each data name in the first list
for data_name in "${data_names1[@]}"; do
    python run_ad_bench.py "$data_name" &
done

wait
