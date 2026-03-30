#!/bin/bash

data_names=("fico" "naticusdroid" "smoking" "TUANDROMD" "airline")


for data_name in "${data_names[@]}"; do
    python3 run_exp_ablation.py "$data_name" "either" "False" &
    python3 run_exp_ablation.py "$data_name" "none" "False" &
    python3 run_exp_ablation.py "$data_name" "either" "True" &
    python3 run_exp_ablation.py "$data_name" "none" "True" &
done

wait

##!/bin/bash
#
#data_names=("fico" "naticusdroid" "smoking" "TUANDROMD" "airline")
#
#run_job() {
#    data_name=$1
#    python3 run_uci.py "$data_name" "either" "False"
#    python3 run_uci.py "$data_name" "none" "False"
#    python3 run_uci.py "$data_name" "either" "True"
#    python3 run_uci.py "$data_name" "none" "True"
#}
#
#export -f run_job
#
## Limit to 2 concurrent jobs, adjust the number after -P as needed
#printf "%s\n" "${data_names[@]}" | xargs -P 2 -n 1 bash -c 'run_job "$@"' _


##!/bin/bash
#
#data_names=("fico" "naticusdroid" "smoking" "TUANDROMD" "airline")
#
#run_job() {
#    data_name=$1
#    python3 run_uci.py "$data_name" "either" "False"
#    python3 run_uci.py "$data_name" "none" "False"
#    python3 run_uci.py "$data_name" "either" "True"
#    python3 run_uci.py "$data_name" "none" "True"
#}
#
#export -f run_job
#
## Run all jobs in parallel, limiting to 2 concurrent jobs at a time
#parallel -j 2 run_job ::: "${data_names[@]}"



