
#! /bin/bash

direc=/home/lujunbo/code/DPFlow/example/deepff/active_learning/c2h6/model_devi/iter_0/01.train
parallel_num=2
run_start=0
run_end=1
parallel_exe=/home/lujunbo/bin/parallel/bin/parallel

seq $run_start $run_end | $parallel_exe -j $parallel_num $direc/produce.sh {} $direc
