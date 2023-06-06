#! /bin/bash

proj_name='ThCl4_aimd'

source /work/chem-lujb/bin/cp2k-8.2/tools/toolchain/install/setup
export PATH=/work/chem-lujb/bin/cp2k-8.2/exe/local:$PATH

direc=`pwd`

for i in {0..9999}
do
cd $direc/data/task_$i
mpirun -np 64 cp2k.popt cp2k.inp 1> cp2k.out 2> cp2k.err
echo $i >> $direc/done_task
done

for i in {0..9999}
do
cd $direc/data/task_$i
if [ ! -s ${proj_name}-1_0.xyz ]
then
echo $i >> $direc/failure_task
mpirun -np 64 cp2k.popt cp2k.inp 1> cp2k.out 2> cp2k.err
fi
done
