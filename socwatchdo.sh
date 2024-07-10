#!/bin/bash

# usage: ./socwatchdo.sh non_modular_snn/snn_model


socwatchdo(){
    sudo /opt/intel/oneapi/vtune/2024.0/socwatch/x64/socwatch -n 1 --feature power --max-detail --result int --output /home/adam/Documents/test_socwatch/$1 --program bash -c "echo $0; echo $1; taskset --cpu-list 1 env MPLBACKEND=Agg /home/adam/mambaforge-pypy3/envs/sinabs/bin/python3 $1.py"
}

socwatchdo $1


