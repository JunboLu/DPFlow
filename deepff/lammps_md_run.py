#! /usr/env/bin python

import os
import math
import time
import subprocess
import numpy as np
from DPFlow.tools import *
from DPFlow.deepff import process
from DPFlow.deepff import gen_lammps_task
from DPFlow.deepff import gen_shell_str

def check_gen_lmp_md(lmp_dir, sys_num, task_num):

  '''
  check_gen_lmp_md: check the status of generating lammps md jobs

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps.
    sys_num: int
      sys_num is the number of lammps systems.
    task_num: int
      task_num is the number of task for each system.
  Returns:
    None
  '''

  check_lmp_md_gen = []
  for i in range(sys_num):
    lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))

    for j in range(task_num):
      lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
      lmp_in_file_name_abs = ''.join((lmp_sys_task_dir, '/md_in.lammps'))
      if ( os.path.exists(lmp_in_file_name_abs) and os.path.getsize(lmp_in_file_name_abs) != 0 ):
        check_lmp_md_gen.append(0)
      else:
        check_lmp_md_gen.append(1)

  if ( len(check_lmp_md_gen) !=0 and all(i == 0 for i in check_lmp_md_gen) ):
    str_print = 'Success: generate lammps molecular dynamics tasks in %s' %(lmp_dir)
    str_print = data_op.str_wrap(str_print, 80, '  ')
    print (str_print, flush=True)
  else:
    log_info.log_error('Generating lammps molecular dynamics tasks error, please check %s' %(lmp_dir))
    exit()

def check_lmp_md_run(lmp_dir, sys_num, task_num):

  '''
  check_lmp_md_run: check the status of running lammps md jobs

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps.
    sys_num: int
      sys_num is the number of lammps systems.
    task_num: int
      task_num is the number of task for each system.
  Returns:
    None
  '''

  check_lmp_md_run = []
  for i in range(sys_num):
    lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))

    cmd = "ls | grep %s" % ('task_')
    task_num = len(call.call_returns_shell(lmp_sys_dir, cmd))

    for j in range(task_num):
      lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
      dump_file_name_abs = ''.join((lmp_sys_task_dir, '/atom.dump'))
      log_file_name_abs = ''.join((lmp_sys_task_dir, '/lammps.out'))
      if ( os.path.exists(dump_file_name_abs) and \
           os.path.getsize(dump_file_name_abs) != 0 and \
           os.path.exists(log_file_name_abs) and \
           file_tools.grep_line_num('Step', log_file_name_abs, lmp_sys_task_dir) != 0 and \
           file_tools.grep_line_num('Loop time', log_file_name_abs, lmp_sys_task_dir) != 0 ):
        check_lmp_md_run.append(0)
      else:
        check_lmp_md_run.append(1)

  if ( len(check_lmp_md_run) != 0 and  all(i == 0 for i in check_lmp_md_run) ):
    print ('  Success: molecular dynamics calculations for %d systems by lammps' %(sys_num))
  else:
    log_info.log_error('Running error: lammps molecular dynamics error, please check %s' %(lmp_dir))
    exit()

def process_md_file(lmp_dir, sys_task_index):


  '''
  process_md_file: process the lammps md files

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps calculation.
    sys_task_index: 2-d int list
      example: [[0,0],[0,1],[1,0],[1,1]]
  Returns:
    None
  '''
 
  for j in range(len(sys_task_index)):
    lmp_sys_task_dir_j = ''.join((lmp_dir, '/sys_', str(sys_task_index[j][0]), '/task_', str(sys_task_index[j][1])))
    file_label = 0
    while True:
      log_file_name = ''.join(('lammps', str(file_label), '.out'))
      log_file_name_abs = ''.join((lmp_sys_task_dir_j, '/', log_file_name))
      dump_file_name = ''.join(('atom', str(file_label), '.dump'))
      dump_file_name_abs = ''.join((lmp_sys_task_dir_j, '/', dump_file_name))
      if ( os.path.exists(log_file_name_abs) and os.path.exists(dump_file_name_abs) \
           and os.path.getsize(dump_file_name_abs) != 0 ):
        file_label = file_label+1
      else:
        file_label = file_label-1
        log_file_name = ''.join(('lammps', str(file_label), '.out'))
        log_file_name_abs = ''.join((lmp_sys_task_dir_j, '/', log_file_name))
        dump_file_name = ''.join(('atom', str(file_label), '.dump'))
        dump_file_name_abs = ''.join((lmp_sys_task_dir_j, '/', dump_file_name))
        break

    if ( os.path.exists(log_file_name_abs) and \
         file_tools.grep_line_num('Step', log_file_name_abs, lmp_sys_task_dir_j) != 0 and \
         file_tools.grep_line_num('Loop time', log_file_name_abs, lmp_sys_task_dir_j) != 0 ):
      gen_lammps_task.combine_frag_traj_file(lmp_sys_task_dir_j)

def lmpmd_single(lmp_dir, sys_index, task_index, lmp_exe, lmp_path, mpi_path, \
                 lmp_mpi_num_per_job, lmp_omp_num_per_job, device, analyze_gpu):

  '''
  lmpmd_single: run single lammps molecular dynamics calculation.

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps calculation.
    sys_index: int
      sys_index is the index of system.
    task_index: int
      task_index is the index of task.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_path: string
      lmp_path is the path of lammps.
    mpi_path: string
      mpi_path is the path of mpi.
    lmp_mpi_num_per_job: int
      lmp_mpi_num_per_job is the mpi number for each lammps job.
    lmp_omp_num_per_job: int
      lmp_omp_num_per_job is the openmp number for each lammps job.
    device: 2-d int list
      device is the name of gpu devices for all nodes.
  Returns:
    none
  '''

  lmp_sys_task_dir = ''.join((lmp_dir, '/sys_', str(sys_index), '/task_', str(task_index)))
  file_label = 0
  while True:
    log_file_name = ''.join(('lammps', str(file_label), '.out'))
    log_file_name_abs = ''.join((lmp_sys_task_dir, '/', log_file_name))
    dump_file_name = ''.join(('atom', str(file_label), '.dump'))
    dump_file_name_abs = ''.join((lmp_sys_task_dir, '/', dump_file_name))
    if ( os.path.exists(log_file_name_abs) and os.path.exists(dump_file_name_abs) and os.path.getsize(dump_file_name_abs) != 0 ):
      file_label = file_label+1
    else:
      break

  if (len(device[0]) == 0) :
    run = '''
#! /bin/bash

lmp_path=%s
mpi_path=%s

export PATH=$lmp_path/bin:$PATH
export PATH=$mpi_path/bin:$PATH
export LD_LIBRARY_PATH=$mpi_path/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=$lmp_path/lib/deepmd_lmp

export OMP_NUM_THREADS=%d

mpirun -np %d %s < ./md_in.lammps 1> %s 2> lammps.err
''' %(lmp_path, mpi_path, lmp_omp_num_per_job, lmp_mpi_num_per_job, lmp_exe, log_file_name)

  elif ( not analyze_gpu and (len(device[0]) > 0) ):
    run = '''
#! /bin/bash

lmp_path=%s

export PATH=$lmp_path/bin:$PATH
export LD_LIBRARY_PATH=$lmp_path/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=$lmp_path/lib/deepmd_lmp

%s < ./md_in.lammps 1> %s 2> lammps.err
''' %(lmp_path, lmp_exe, log_file_name)

  elif ( analyze_gpu and (len(device[0]) > 0) ):
    device_str=data_op.comb_list_2_str(device[0], ',')
    run = '''
#! /bin/bash

lmp_path=%s
mpi_path=%s

export PATH=$lmp_path/bin:$PATH
export PATH=$mpi_path/bin:$PATH
export LD_LIBRARY_PATH=$mpi_path/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=$lmp_path/lib/deepmd_lmp

export CUDA_VISIBLE_DEVICES=%s
export OMP_NUM_THREADS=%d

mpirun -np %d %s < ./md_in.lammps 1> %s 2> lammps.err
''' %(lmp_path, mpi_path, device_str, lmp_omp_num_per_job, lmp_mpi_num_per_job, lmp_exe, log_file_name)

  run_file_name_abs = ''.join((lmp_sys_task_dir, '/run.sh'))
  with open(run_file_name_abs, 'w') as f:
    f.write(run)

  subprocess.run('chmod +x run.sh', cwd=lmp_sys_task_dir, shell=True)
  try:
    subprocess.run("bash -c './run.sh'", cwd=lmp_sys_task_dir, shell=True)
  except subprocess.CalledProcessError as err:
    log_info.log_error('Running error: %s command running error in %s' %(err.cmd, lmp_sys_task_dir))
    exit()

  if ( os.path.exists(log_file_name_abs) and \
       file_tools.grep_line_num('Step', log_file_name_abs, lmp_sys_task_dir) != 0 and \
       file_tools.grep_line_num('Loop time', log_file_name_abs, lmp_sys_task_dir) != 0 ):
    gen_lammps_task.combine_frag_traj_file(lmp_sys_task_dir)

def lmpmd_parallel(lmp_dir, lmp_path, mpi_path, lmp_exe, parallel_exe, sys_index_str, \
                   task_index_str, mpi_num_str, device_num_str, device_id_start_str, device, \
                   lmp_md_job_per_node, lmp_omp_num_per_job, proc_num_per_node, ssh, host, analyze_gpu):

  '''
  lmpmd_parallel: run lammps molecular dynamics calculation in parallel.

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps calculation.
    lmp_path: string
      lmp_path is the path of lammps.
    mpi_path: string
      mpi_path is the path of mpi.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    parallel_exe: string
      parallel_exe is the parallel executable file.
    sys_index_str: string
      sys_index_str is the string containing systems index.
    task_index_str: string
      task_index_str is the string containing tasks index.
    mpi_num_str: string
      mpi_num_str is the string containing mpi number.
    device_num_str: string
      device_num_str is the string containing gpu devices number.
    device_id_start_str: string
      device_id_start_str is the string containing staring gpu device id.
    lmp_md_job_per_node: int
      lmp_md_job_per_node is the number of lammps job in one node.
    proc_num_per_node: 1-d int list
      proc_num_per_node is the number of processors in each node.
    ssh: bool
      ssh is whether to ssh to computational node.
    host: 1-d string list
      host is the name of computational nodes.
  Returns:
    none
  '''

  host_name_proc = []
  for l in range(len(host)):
    host_name_proc.append(''.join((str(proc_num_per_node[l]), '/', host[l])))
  host_info = data_op.comb_list_2_str(host_name_proc, ',')

  device_str = data_op.comb_list_2_str(device[0], ' ')

  run_1 = '''
#! /bin/bash

sys_job="%s"
task_job="%s"
mpi_num="%s"
device_num="%s"
device_start="%s"
device="%s"
direc=%s
parallel_exe=%s

sys_job_arr=(${sys_job///})
task_job_arr=(${task_job///})
mpi_num_arr=(${mpi_num///})
device_num_arr=(${device_num///})
device_start_arr=(${device_start///})

num=${#sys_job_arr[*]}

for ((i=0;i<=num-1;i++));
do
sys_task_mpi_num_arr[i]="${sys_job_arr[i]} ${task_job_arr[i]} ${mpi_num_arr[i]} ${device_num_arr[i]} ${device_start_arr[i]} $device"
done
''' %(sys_index_str, task_index_str, mpi_num_str, device_num_str, device_id_start_str, device_str, lmp_dir, parallel_exe)
  if ssh:
    run_2 = '''
for i in "${sys_task_mpi_num_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --controlmaster -S %s --sshdelay 0.2 $direc/produce.sh {} $direc
''' %(lmp_md_job_per_node, host_info)
  else:
    run_2 = '''
for i in "${sys_task_mpi_num_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --delay 0.2 $direc/produce.sh {} $direc
''' %(lmp_md_job_per_node)

  produce_1 = '''
#! /bin/bash

x=$1
direc=$2

x_arr=(${x///})

new_direc=$direc/sys_${x_arr[0]}/task_${x_arr[1]}

lmp_path=%s
mpi_path=%s

ulimit -u 204800
export PATH=$lmp_path/bin:$PATH
export PATH=$mpi_path/bin:$PATH
export LD_LIBRARY_PATH=$mpi_path/lib:$LD_LIBRARY_PATH
export LAMMPS_PLUGIN_PATH=$lmp_path/lib/deepmd_lmp
''' %(lmp_path, mpi_path)

  produce_2 = '''
device_num=${x_arr[3]}
if [ $device_num != 0 ]; then
device_id_start=${x_arr[4]}
for ((i=0;i<=device_num-1;i++));
do
((id=$device_id_start+$i+5))
m[i]=${x_arr[id]}
done

for ((i=0;i<device_num;i++));
do
str=$str${m[i]}","
done

export CUDA_VISIBLE_DEVICES=${str:0:(2*$device_num-1)}
fi
'''

  produce_3 = '''
a=0

while true
do
if [[ -f $new_direc/atom$a.dump && -f $new_direc/lammps$a.out ]]
then
((a=$a+1))
else
break
fi
done
'''

  produce_4 = '''
cd $new_direc
%s < ./md_in.lammps 1> lammps$a.out 2> lammps.err
cd $direc
''' %(lmp_exe)

  produce_5 = '''
export OMP_NUM_THREADS=%d 
cd $new_direc
mpirun -np ${x_arr[2]} %s < ./md_in.lammps 1> lammps$a.out 2> lammps.err
cd $direc
''' %(lmp_omp_num_per_job, lmp_exe)

  run_file_name_abs = ''.join((lmp_dir, '/run.sh'))
  with open(run_file_name_abs, 'w') as f:
    f.write(run_1+run_2)

  produce_file_name_abs = ''.join((lmp_dir, '/produce.sh'))
  with open(produce_file_name_abs, 'w') as f:
    if ( analyze_gpu and len(device[0]) > 0 ):
      f.write(produce_1+produce_2+produce_3+produce_4)
    elif ( len(device[0]) == 0 ):
      f.write(produce_1+produce_3+produce_5)
    elif ( not analyze_gpu and len(device[0]) > 0 ):
      f.write(produce_1+produce_3+produce_4)

  subprocess.run('chmod +x run.sh', cwd=lmp_dir, shell=True)
  subprocess.run('chmod +x produce.sh', cwd=lmp_dir, shell=True)
  try:
    subprocess.run("bash -c './run.sh'", cwd=lmp_dir, shell=True)
  except subprocess.CalledProcessError as err:
    log_info.log_error('Running error: %s command running error in %s' %(err.cmd, lmp_dir))
    exit()

def run_lmpmd_ws(work_dir, iter_id, lmp_path, lmp_exe, parallel_exe, mpi_path, lmp_md_job_per_node, \
                 lmp_mpi_num_per_job, lmp_omp_num_per_job, proc_num_per_node, host, device, analyze_gpu):

  '''
  rum_lmpmd_ws: kernel function to run lammps md for workstation mode.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    lmp_path: string
      lmp_path is the path of lammps.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    parallel_exe: string
      parallel_exe is the parallel executable file.
    mpi_path: string
      mpi_path is the path of mpi.
    lmp_md_job_per_node: int
      lmp_md_job_per_node is the number of lammps job in one node.
    lmp_mpi_num_per_job: int
      lmp_mpi_num_per_job is the mpi number for each lammps job.
    lmp_omp_num_per_job: int
      lmp_omp_num_per_job is the openmp number for each lammps job.
    proc_num_per_node: 1-d int list
      proc_num_per_node is the number of processors in each node.
    host: 1-d string list
      host is the name of computational nodes.
    device: 2-d int list
      device is the name of gpu devices for all nodes.
  Returns:
    none
  '''

  lmp_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))

  cmd = "ls | grep %s" % ('sys_')
  sys_num = len(call.call_returns_shell(lmp_dir, cmd))

  #All systems have same number of tasks
  lmp_sys_0_dir = ''.join((lmp_dir, '/sys_0'))
  cmd = "ls | grep %s" % ('task_')
  task_num = len(call.call_returns_shell(lmp_sys_0_dir, cmd))

  #check generating lammps tasks
  check_gen_lmp_md(lmp_dir, sys_num, task_num)

  #run lammps md
  if ( sys_num == 1 and task_num == 1 ):
    lmpmd_single(lmp_dir, 0, 0, lmp_exe, lmp_path, mpi_path, lmp_mpi_num_per_job, \
                 lmp_omp_num_per_job, device, analyze_gpu)
  else:
    total_task_num = sys_num*task_num
    sys_task_index = []
    for i in range(sys_num):
      for j in range(task_num):
        sys_task_index.append([i,j])
    calculated_num = 0
    for i in range(total_task_num):
      lmp_sys_task_dir = ''.join((lmp_dir, '/sys_', str(sys_task_index[i][0]), '/task_', str(sys_task_index[i][1])))
      log_file_name = ''.join((lmp_sys_task_dir, '/lammps.out'))
      dump_file_name = ''.join((lmp_sys_task_dir, '/atom.dump'))
      if ( os.path.exists(log_file_name) and os.path.exists(dump_file_name) ):
        calculated_num = calculated_num + 1
      else:
        break
    device_num = len(device[0])
    if ( device_num > 0 and device_num < lmp_md_job_per_node ):
      lmp_md_job_per_node = device_num

    if ( len(host) > 1 and all(len(i) == 0 for i in device) ):
      ssh = True
    else:
      ssh = False

    if ( calculated_num < total_task_num ):
      run_start = calculated_num
      run_end = run_start+lmp_md_job_per_node*len(host)-1
      if ( run_end > total_task_num-1 ):
        run_end=total_task_num-1
      cycle = math.ceil((total_task_num-calculated_num)/(lmp_md_job_per_node*len(host)))
      for i in range(cycle):
        if ( device_num > 0 ):
          device_num_list = data_op.int_split(device_num, lmp_md_job_per_node)
        else:
          device_num_list = [0]*lmp_md_job_per_node*len(host)
        device_id_start = [0]
        for j in range(len(device_num_list)-1):
          device_id_start.append(device_id_start[j]+device_num_list[j])
        device_num_str = data_op.comb_list_2_str(device_num_list[0:(run_end-run_start+1)], ' ')
        device_id_start_str = data_op.comb_list_2_str(device_id_start[0:(run_end-run_start+1)], ' ')

        tot_mpi_num_list = []
        for proc_num in proc_num_per_node:
          mpi_num_list = [lmp_mpi_num_per_job]*lmp_md_job_per_node
          tot_mpi_num_list.append(mpi_num_list)
        tot_mpi_num_list = data_op.list_reshape(tot_mpi_num_list)[0:(run_end-run_start+1)]
        mpi_num_str = data_op.comb_list_2_str(tot_mpi_num_list, ' ')
        sys_task_index_part = sys_task_index[run_start:run_end+1]
        sys_index = [sys_task[0] for sys_task in sys_task_index_part]
        task_index = [sys_task[1] for sys_task in sys_task_index_part]
        sys_index_str = data_op.comb_list_2_str(sys_index, ' ')
        task_index_str = data_op.comb_list_2_str(task_index, ' ')

        lmpmd_parallel(lmp_dir, lmp_path, mpi_path, lmp_exe, parallel_exe, sys_index_str, \
                       task_index_str, mpi_num_str, device_num_str, device_id_start_str, \
                       device, lmp_md_job_per_node, lmp_omp_num_per_job, proc_num_per_node, ssh, host, analyze_gpu)

        run_start = run_start + lmp_md_job_per_node*len(host)
        run_end = run_end + lmp_md_job_per_node*len(host)
        if ( run_end > total_task_num-1):
          run_end = total_task_num-1
    else:
      lmpmd_single(lmp_dir, sys_task_index[calculated_id][0], sys_task_index[calculated_id][1], \
                   lmp_exe, lmp_path, mpi_path, lmp_mpi_num_per_job, lmp_omp_num_per_job, device)

    process_md_file(lmp_dir, sys_task_index)

  #check lammps md
  check_lmp_md_run(lmp_dir, sys_num, task_num)

def run_lmpmd_as(work_dir, iter_id, lmp_queue, max_lmp_job, lmp_core_num, lmp_gpu_num, \
                 submit_system, lmp_path, lmp_exe, mpi_path, analyze_gpu):

  '''
  run_lmpmd_as: kernel function to run lammps md for auto_submit mode.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    lmp_queue: string
      lmp_queue is the queue name of lammps job.
    max_lmp_job: int
      max_lmp_job is the maximum number of dp job.
    lmp_core_num: int
      lmp_core_num is the number of cores for each lammps md job.
    lmp_gpu_num: int
      lmp_gpu_num is the number of gpus for each lammps md job.
    submit_system: string
      submit_system is the submition system.
    lmp_path: string
      lmp_path is the path of lammps.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    mpi_path: string
      mpi_path is the path of mpi.
    analyze_gpu: bool
      analyze_gpu is whether program need to analyze GPU.
  Returns:
    none
  '''

  from DPFlow.deepff import gen_shell_str

  lmp_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))

  cmd = "ls | grep %s" % ('sys_')
  sys_num = len(call.call_returns_shell(lmp_dir, cmd))

  #All systems have same number of tasks
  lmp_sys_0_dir = ''.join((lmp_dir, '/sys_0'))
  cmd = "ls | grep %s" % ('task_')
  task_num = len(call.call_returns_shell(lmp_sys_0_dir, cmd))

  #check generating lammps tasks
  check_gen_lmp_md(lmp_dir, sys_num, task_num)

  total_task_num = sys_num*task_num
  sys_task_index = []
  for i in range(sys_num):
    for j in range(task_num):
      sys_task_index.append([i,j])
  calculated_num = 0
  for i in range(total_task_num):
    lmp_sys_task_dir = ''.join((lmp_dir, '/sys_', str(sys_task_index[i][0]), '/task_', str(sys_task_index[i][1])))
    log_file_name = ''.join((lmp_sys_task_dir, '/lammps.out'))
    dump_file_name = ''.join((lmp_sys_task_dir, '/atom.dump'))
    if ( os.path.exists(log_file_name) and os.path.exists(dump_file_name) ):
      calculated_num = calculated_num+1
    else:
      break

  if ( max_lmp_job > total_task_num ):
    max_lmp_job = total_task_num
  lmp_queue = lmp_queue*max_lmp_job
  lmp_queue_sub = lmp_queue[0:max_lmp_job]
  cycle = math.ceil((total_task_num-calculated_num)/max_lmp_job)
  for i in range(cycle):
    lmp_md_id_cycle = []
    rand_cycle = []
    for j in range(max_lmp_job):
      lmp_md_id = calculated_num+i*max_lmp_job+j
      if ( lmp_md_id < total_task_num ):
        rand_int = np.random.randint(10000000000)
        rand_cycle.append(rand_int)
        lmp_md_id_cycle.append(lmp_md_id)
        lmp_sys_task_dir = ''.join((lmp_dir, '/sys_', str(sys_task_index[lmp_md_id][0]), \
                                    '/task_', str(sys_task_index[lmp_md_id][1])))
        job_label = ''.join(('lmp_md_', str(rand_int)))
        flag_file_name_abs = ''.join((lmp_sys_task_dir, '/success.flag'))
        if ( os.path.exists(flag_file_name_abs) ):
          subprocess.run('rm %s' %(flag_file_name_abs), cwd=lmp_sys_task_dir, shell=True)
        if ( submit_system == 'lsf' ):
          submit_file_name_abs = ''.join((lmp_sys_task_dir, '/lmp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( lmp_gpu_num > 0 and not analyze_gpu ):
              script_1 = gen_shell_str.gen_lsf_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num, lmp_core_num)
              script_3 = gen_shell_str.gen_cd_lsfcwd()
              script_4 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_5 = gen_shell_str.gen_lmp_file_label()
              script_6 = gen_shell_str.gen_lmp_gpu_cmd(lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( lmp_gpu_num > 0 and analyze_gpu ):
              script_1 = gen_shell_str.gen_lsf_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num, lmp_core_num)
              script_3 = gen_shell_str.gen_cd_lsfcwd()
              script_4 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_5 = gen_shell_str.gen_lmp_file_label()
              script_6 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
              script_7 = gen_shell_str.gen_lmp_gpu_cmd(lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6+script_7)

            if ( lmp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_lsf_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_cd_lsfcwd()
              script_3 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_4 = gen_shell_str.gen_lmp_file_label()
              script_5 = gen_shell_str.gen_lmp_cpu_cmd(lmp_core_num, lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5)

          subprocess.run('bsub < ./lmp.sub', cwd=lmp_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

        if ( submit_system == 'pbs' ):
          submit_file_name_abs = ''.join((lmp_sys_task_dir, '/lmp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( lmp_gpu_num > 0 ):
              script_1 = gen_shell_str.gen_pbs_normal(lmp_queue_sub[j], lmp_core_num, lmp_gpu_num, job_label)
              script_2 = gen_shell_str.gen_cd_pbscwd()
              script_3 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_4 = gen_shell_str.gen_lmp_file_label()
              script_5 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
              script_6 = gen_shell_str.gen_lmp_gpu_cmd(lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( lmp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_pbs_normal(lmp_queue_sub[j], lmp_core_num, lmp_gpu_num, job_label)
              script_2 = gen_shell_str.gen_cd_pbscwd()
              script_3 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_4 = gen_shell_str.gen_lmp_file_label()
              script_5 = gen_shell_str.gen_lmp_cpu_cmd(lmp_core_num, lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5)

          subprocess.run('qsub < ./lmp.sub', cwd=lmp_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

        if ( submit_system == 'slurm' ):
          submit_file_name_abs = ''.join((lmp_sys_task_dir, '/lmp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( lmp_gpu_num > 0 and not analyze_gpu ):
              script_1 = gen_shell_str.gen_slurm_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_slurm_gpu_set(lmp_gpu_num)
              script_3 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_4 = gen_shell_str.gen_lmp_file_label()
              script_5 = gen_shell_str.gen_lmp_gpu_cmd(lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5)

            if ( lmp_gpu_num > 0 and analyze_gpu ):
              script_1 = gen_shell_str.gen_slurm_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_slurm_gpu_set(lmp_gpu_num)
              script_3 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_4 = gen_shell_str.gen_lmp_file_label()
              script_5 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
              script_6 = gen_shell_str.gen_lmp_gpu_cmd(lmp_exe)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( lmp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_slurm_normal(lmp_queue_sub[j], lmp_core_num, job_label)
              script_2 = gen_shell_str.gen_lmp_env(lmp_path, mpi_path)
              script_3 = gen_shell_str.gen_lmp_file_label()
              script_4 = gen_shell_str.gen_lmp_cpu_cmd(lmp_core_num, lmp_exe)
              f.write(script_1+script_2+script_3+script_4)

          subprocess.run('sbatch ./lmp.sub', cwd=lmp_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

    job_id = []
    failure_id = []
    for j in range(len(lmp_md_id_cycle)):
      job_id_j = process.get_job_id(work_dir, submit_system, 'lmp_md_', rand_cycle[j])
      if ( job_id_j > 0 ):
        job_id.append(job_id_j)
      else:
        failure_id.append(lmp_md_id_cycle[j])
    if ( len(job_id) == len(lmp_md_id_cycle) ):
      for j in range(len(lmp_md_id_cycle)):
        lmp_md_id = lmp_md_id_cycle[j]
        str_print = 'Success: submit lammps md job for system %d task %d in iteration %d with job id %d' \
                     %(sys_task_index[lmp_md_id][0], sys_task_index[lmp_md_id][1], iter_id, job_id[j])
        str_print = data_op.str_wrap(str_print, 80, '  ')
        print (str_print, flush=True)
      while True:
        time.sleep(10)
        judge = []
        for j in lmp_md_id_cycle:
          flag_file_name = ''.join((lmp_dir, '/sys_', str(sys_task_index[j][0]), \
                                    '/task_', str(sys_task_index[j][1]), '/success.flag'))
          judge.append(os.path.exists(flag_file_name))
        if all(judge):
          break
    else:
      for j in lmp_md_id_cycle:
        log_info.log_error('Fail to submit lammps md job for system %d task %d in iteration %d' \
                           %(sys_task_index[j][0], sys_task_index[j][1], iter_id))
      exit()

  process_md_file(lmp_dir, sys_task_index)

  #check lammps md
  check_lmp_md_run(lmp_dir, sys_num, task_num)
 
if __name__ == '__main__':
  from DPFlow.tools import read_input
  from DPFlow.deepff import lammps_run
  from DPFlow.deepff import check_deepff

  box_file = '/home/lujunbo/WORK/Deepmd/DPFlow/co2/md/lmp_init_data/box'
  coord_file = '/home/lujunbo/WORK/Deepmd/DPFlow/co2/md/lmp_init_data/str.inc'
  tri_cell_vec, atoms, x, y, z = get_box_coord(box_file, coord_file)
  print (atoms, x, y, z)

  exit()
  work_dir = '/home/lujunbo/code/github/DPFlow/deepff/work_dir'
  deepff_key = ['deepmd', 'lammps', 'cp2k', 'model_devi', 'environ']
  deepmd_dic, lmp_dic, cp2k_dic, model_devi_dic, environ_dic = \
  read_input.dump_info(work_dir, 'input.inp', deepff_key)
  proc_num = 4
  deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic = \
  check_deepff.check_inp(deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic, proc_num)

  #Test gen_lmpmd_task function
  atoms_type_multi_sys, atoms_num_tot = \
  lmp_run.gen_lmpmd_task(lmp_dic, work_dir, 0)
  #print (atoms_type_multi_sys, atoms_num_tot)

  #Test run_lmpmd function
  lmp_run.run_lmpmd(work_dir, 0, 4)

  #Test gen_lmpfrc_file function
  atoms_num_tot = {0:3}
  atoms_type_multi_sys = {0: {'O': 1, 'H': 2}}
  lmp_run.gen_lmpfrc_file(work_dir, 0, atoms_num_tot, atoms_type_multi_sys)

  #Test run_lmpfrc
  lmp_run.run_lmpfrc(work_dir, 0, 4)
