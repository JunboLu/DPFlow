
#! /bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

dp_path=/home/lujunbo/bin/deepmd-kit-2.2.2-cpu

export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH

x=$1
direc=$2
cd $direc/$x
dp train input.json 1> log.out 2> log.err
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
cd /home/lujunbo/code/DPFlow/example/deepff/active_learning/c2h6/model_devi
