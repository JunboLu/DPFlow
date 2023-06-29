#! /env/bin/python

def gen_lsf_normal(queue, core_num, job_name):

  '''
  gen_lsf_normal: generate normal input for lsf submit script

  Args:
    queue: string
      queue is the name of queue.
    core_num: int
      core_num is the number of cpu cores.
    job_name: string
      job_name is the name of job.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
#!/bin/bash
#BSUB -q %s
#BSUB -n %d
#BSUB -J %s
#BSUB -R "span[hosts=1]"
#BSUB -e %s.err
#BSUB -o %s.out
''' %(queue, core_num, job_name, '%J', '%J')

  return script

def gen_pbs_normal(queue, core_num, gpu_num, job_name):

  '''
  gen_pbs_normal: generate normal input for pbs submit script

  Args:
    queue: string
      queue is the name of queue.
    core_num: int
      core_num is the number of cpu cores.
    job_name: string
      job_name is the name of job.
  Return:
    script: string
      script is the returning script.
  '''

  if ( gpu_num > 0 ):
    script = '''
#!/bin/bash
#PBS -q %s
#PBS -l nodes=1:ppn=%d:gpu=%d
#PBS -j oe
#PBS -V
#PBS -N %s
#PBS -o output.o
#PBS -e error.e
''' %(queue, core_num, gpu_num, job_name)
  else:
    script = '''
#!/bin/bash
#PBS -q %s
#PBS -l nodes=1:ppn=%d
#PBS -j oe
#PBS -V
#PBS -N iter_%d_%s
#PBS -o output.o
#PBS -e error.e
''' %(queue, core_num, job_name)

  return script

def gen_slurm_normal(core_num, queue, job_name):

  '''
  gen_pbs_normal: generate normal input for pbs submit script

  Args:
    queue: string
      queue is the name of queue.
    core_num: int
      core_num is the number of cpu cores.
    job_name: string
      job_name is the name of job.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
#!/bin/bash

#SBATCH -n %d
#SBATCH -p %s
#SBATCH --job-name=iter_%d_%s
#SBATCH -o output.o
#SBATCH -e error.e
''' %(core_num, queue, job_name)

  return script

def gen_slurm_gpu_set(gpu_num):

  '''
  gen_slurm_gpu_set: generate input for lsf gpu setting

  Args:
    gpu_num: int
      gpu_num is the number of gpus.
    core_num: int
      core_num is the number of cpu cores.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
#SBATCH --gres=gpu:%d
''' %(gpu_num)

  return script

def gen_lsf_gpu_set(gpu_num, core_num):

  '''
  gen_lsf_gpu_set: generate input for lsf gpu setting

  Args:
    gpu_num: int
      gpu_num is the number of gpus.
    core_num: int
      core_num is the number of cpu cores.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
#BSUB -gpu "num=%d/host"
#BSUB -R "affinity[core(%d)]"
''' %(gpu_num, core_num)

  return script

def gen_cd_lsfcwd():

  '''
  gen_cd_lsfcwd: generate input for cd lsf working directory

  Args:
    None
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
cd $LS_SUBCWD
'''

  return script

def gen_cd_pbscwd():

  '''
  gen_cd_pbscwd: generate input for cd pbs working directory

  Args:
    None
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
cd $PBS_O_WORKDIR
'''

  return script

def gen_lmp_env(lmp_path, mpi_path):

  '''
  gen_lmp_env: generate environment setting for lammps

  Args:
    lmp_path: string
      lmp_path is the path of lammps.
    mpi_path: string
      mpi_path is the path of mpi.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
lmp_path=%s
mpi_path=%s

export PATH=$lmp_path/bin:$PATH
export PATH=$mpi_path/bin:$PATH
export LD_LIBRARY_PATH=$mpi_path/lib:$LD_LIBRARY_PATH
''' %(lmp_path, mpi_path)

  return script

def gen_dp_env(dp_path):

  '''
  gen_dp_env: generate environment setting for deepmd-kit

  Args:
    dp_path: string
      dp_path is the path of deepmd-kit.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
dp_path=%s
export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH
''' %(dp_path)

  return script

def gen_cuda_env(cuda_dir):

  '''
  gen_cuda_env: generate environment setting for cuda

  Args:
    cuda_dir: string
      cuda_dir is the path of cuda.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
CUDA_DIR=%s
export PATH=$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH

export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
''' %(cuda_dir)

  return script

def gen_gpu_analyze(gpu_num):

  '''
  gen_gpu_analyze: generate shell code for analyzing gpu

  Args:
    gpu_num: int
      gpu_num is the number of gpus.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
gpu_id=-1

find_gpu()
{
  res=$(nvidia-smi | \
    grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
    awk '{print ($9" "$11)}' | \
    sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
    awk '{print $2 - $1}')

    i=0
    res=($(for s in $res; do echo $i $s && i=`expr 1 + $i`; done | \
        sort -n -k 2 -r))

    n_gpu_req=${1-"1"}
    mem_lb=${2-"0"}

  gpu_id=-1
  n=0
  for i in $(seq 0 2 `expr ${#res[@]} - 1`); do
    gid=${res[i]}
    mem=${res[i+1]}
    if [ $n -lt ${n_gpu_req} -a $mem -ge ${mem_lb} ]; then
      if [ $n -eq 0 ]; then
          gpu_id=$gid
      else
          gpu_id=${gpu_id}","$gid
      fi
        n=`expr 1 + $n`
    else
      break
    fi
  done

find_gpu %d
export CUDA_VISIBLE_DEVICES=$gpu_id
''' %(gpu_num)

  return script

def gen_lmp_file_label():

  '''
  gen_lmp_file_label: generate shell code for analyzing lammps output file label

  Args:
    None
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
a=0

while true
do
if [[ -f atom$a.dump && -f lammps$a.out ]]
then
((a=$a+1))
else
break
fi
done
'''

  return script

def gen_dp_cmd(dp_cmd):

  '''
  gen_dp_cmd: generate shell code for deepmd-kit command

  Args:
    dp_cmd: string
      dp_cmd is the deepmd-kit command.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
if [ -f "success.flag" ]; then
rm success.flag
fi

%s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
echo 'success' > success.flag
''' %(dp_cmd)

  return script

def gen_lmp_cpu_cmd(core_num, lmp_exe):

  '''
  gen_lmp_cpu_cmd: generate shell code for deepmd-kit command using cpu

  Args:
    core_num: int
      core_num is the number of cpu cores.
    lmp_exe: string
      lmp_exe is the lammps executable file.
  Return:
    script: string
      script is the returning script.
  '''

  script_ = '''
if [ -f "success.flag" ]; then
rm success.flag
fi

export OMP_NUM_THREADS=2
mpirun -np %d %s < ./md_in.lammps 1> lammps$a.out 2> lammps.err

echo 'success' > success.flag
''' %(core_num/2, lmp_exe)

  return script

def gen_lmp_gpu_cmd(lmp_exe):

  '''
  gen_lmp_gpu_cmd: generate shell code for deepmd-kit command using gpu

  Args:
    lmp_exe: string
      lmp_exe is the lammps executable file.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
if [ -f "success.flag" ]; then
rm success.flag
fi

%s < ./md_in.lammps 1> lammps$a.out 2> lammps.err

echo 'success' > success.flag
''' %(lmp_exe)

  return script

def gen_lmp_frc_dis(model_dir, task_index, parallel_exe, core_num):

  '''
  gen_lmp_frc_dis: generate shell code for lammps discrete force jobs

  Args:
    model_dir: string
      model_dir is the lammps force directory for one model.
    task_index: 1-d int list
      task_index is the index of tasks.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    core_num: int
      core_num is the number of cpu cores.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
direc=%s
task_index="%s"
parallel_exe=%s

ulimit -u 65535

task_index_arr=(${task_index///})
num=${#task_index_arr[*]}

for i in "${task_index_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --delay 0.2 $direc/produce.sh {} $direc
''' %(model_dir, task_index, parallel_exe, core_num)

  return script

def gen_lmq_frc_ser(model_dir, parallel_exe, start, end, core_num):

  '''
  gen_lmp_frc_ser: generate shell code for lammps serial force jobs

  Args:
    model_dir: string
      model_dir is the lammps force directory for one model.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    start: int
      start is the starting index.
    end: int
      end is the ending index.
    core_num: int
      core_num is the number of cpu cores.
  Return:
    script: string
      script is the returning script.
  '''

  import math

  script = '''
direc=%s
parallel_exe=%s
start=%d
end=%d
parallel_num=%d

ulimit -u 65535

a=$start
((b=$a+$parallel_num-1))

if [ $b -gt $end ]; then
b=$end
fi

for i in {1..%d}
do
for j in $(seq $a $b); do echo "$j"; done | $parallel_exe -j %d --delay 0.2 $direc/produce.sh {} $direc
((a=$a+$parallel_num))
((b=$b+$parallel_num))
if [ $b -gt $end ]; then
b=$end
fi
done
''' %(model_dir, parallel_exe, start, end, core_num, math.ceil((end-start+1)/core_num), core_num)

  return script

def gen_cp2k_script(set_cp2k_env, cp2k_sys_task_dir, task_index, core_num, cp2k_exe):

  '''
  gen_cp2k_script: generate shell code for cp2k force jobs

  Args:
    set_cp2k_env: string
      set_cp2k_env is the command for setting cp2k environment.
    cp2k_sys_task_dir: string
      cp2k_sys_task_dir is the cp2k directory.
    task_index: 1-d int list
      task_index is the index of cp2k tasks.
    core_num: int
      core_num is the number of cpu cores.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
  Return:
    script: string
      script is the returning script.
  '''

  script = '''
%s
direc=%s

for i in %s
do
new_direc=$direc/traj_$i
cd $new_direc
RESTART=`ls | grep 'RESTART.wfn'`
if [ $? -eq 0 ]; then
rm *RESTART.wfn*
fi
if [ -f "success.flag" ]; then
rm success.flag
fi
if [ -f "cp2k-1_0.xyz" ]; then
rm cp2k-1_0.xyz
fi
if [ -f "cp2k-1.coordLog" ]; then
rm cp2k-1.coordLog
fi
if [ -f "cp2k-1.Log" ]; then
rm cp2k-1.Log
fi
mpirun -np %d %s $new_direc/input.inp 1> $new_direc/cp2k.out 2> $new_direc/cp2k.err
converge_info=`grep "SCF run NOT converged" cp2k.out`
if [ $? -eq 0 ]; then
if [ -f "cp2k-1_0.xyz" ]; then
rm cp2k-1_0.xyz
fi
if [ -f "cp2k-1.coordLog" ]; then
rm cp2k-1.coordLog
fi
if [ -f "cp2k-1.Log" ]; then
rm cp2k-1.Log
fi
mpirun -np %d %s $new_direc/input.inp 1> $new_direc/cp2k.out 2> $new_direc/cp2k.err
fi
echo 'success' > success.flag
done
''' %(set_cp2k_env, cp2k_sys_task_dir, task_index, core_num, cp2k_exe, core_num, cp2k_exe)

  return script
