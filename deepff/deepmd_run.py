#! /usr/env/bin python

import os
import math
import json
import subprocess
import linecache
import numpy as np
from collections import OrderedDict
from DPFlow.tools import file_tools
from DPFlow.tools import call
from DPFlow.tools import log_info
from DPFlow.tools import data_op
from DPFlow.deepff import process
from DPFlow.deepff import gen_shell_str

def pre_process(work_dir, iter_id, use_prev_model, start, end, dp_version, base):

  '''
  pre_process: pre_process the dp job

  Args:
    work_dir: string
      work_dir is the working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    use_prev_model: bool
      use_prev_model is whether we need to use previous model.
    start: int
      start is the starting model id.
    end: int
      end is the endding model id.
  Returns:
    dp_cmd: string
      dp_cmd is the command of dp job.
  '''

  model_list = list(range(start, end+1, 1))

  deepmd_train_dir = ''.join((work_dir, '/iter_', str(iter_id), '/01.train'))
  model_ckpt_file_exists = []
  lcurve_file_exists = []
  final_batch = []
  for i in model_list:
    if ( i == 0 ):
      input_file = ''.join((deepmd_train_dir, '/', str(i), '/input.json'))
      with open(input_file, 'r') as f:
        deepmd_dic = json.load(f)
      save_freq = deepmd_dic['training']['save_freq']
    model_ckpt_file = ''.join((deepmd_train_dir, '/', str(i), '/model.ckpt.index'))
    lcurve_file = ''.join((deepmd_train_dir, '/', str(i), '/lcurve.out'))
    model_ckpt_file_exists.append(os.path.exists(model_ckpt_file))
    lcurve_file_exists.append(os.path.exists(lcurve_file))
    if ( os.path.exists(lcurve_file) ):
      whole_line_num = len(open(lcurve_file, 'r').readlines())
      if ( dp_version == '1.3.3' ):
        batch_line = file_tools.grep_line_num('batch', lcurve_file, work_dir)
        max_print_num = 8
      else:
        batch_line = file_tools.grep_line_num('step', lcurve_file, work_dir)
        max_print_num = 5
      if ( whole_line_num > len(batch_line) ):
        line_num = whole_line_num
        while True:
          line = linecache.getline(lcurve_file, line_num)
          line_split = data_op.split_str(line, ' ', '\n')
          if ( len(line_split) >= max_print_num and data_op.eval_str(line_split[0]) == 1 ):
            break
          line_num = line_num-1
      else:
        final_batch.append(0)
      line = linecache.getline(lcurve_file, line_num)
      linecache.clearcache()
      line_split = data_op.split_str(line, ' ', '\n')
      if ( len(line_split) >= max_print_num ):
        final_batch.append(int(line_split[0]))
      else:
        final_batch.append(0)
    else:
      final_batch.append(0)

  if ( all(file_exist for file_exist in model_ckpt_file_exists) and \
       all(file_exist for file_exist in lcurve_file_exists) and \
       all(i > save_freq for i in final_batch) ):
    dp_cmd = 'dp train --restart model.ckpt input.json 1>> log.out 2>> log.err'
  else:
    if ( use_prev_model and iter_id>base ):
      for i in model_list:
        model_dir = ''.join((deepmd_train_dir, '/', str(i)))
        prev_model_file = ''.join((work_dir, '/iter_', str(iter_id-1), '/01.train/', str(i), '/model.ckpt.*'))
        cmd = "cp %s %s" %(prev_model_file, model_dir)
        call.call_simple_shell(model_dir, cmd)
      dp_cmd = 'dp train --init-model model.ckpt input.json 1> log.out 2> log.err'
    else:
      dp_cmd = 'dp train input.json 1> log.out 2> log.err'

  return dp_cmd

def check_gen_deepmd(train_dir, model_num):

  '''
  check_gen_deepmd: check the status of generating dp jobs

  Args:
    train_dir: string
      train_dir is the directory of dp train.
    model_num: int
      model_num is the number of dp train models.
  Returns:
    None
  '''

  check_deepmd_gen = []
  for i in range(model_num):
    inp_file_name_abs = ''.join((train_dir, '/', str(i), '/input.json'))
    if ( os.path.exists(inp_file_name_abs) and os.path.getsize(inp_file_name_abs) != 0 ):
      check_deepmd_gen.append(0)
    else:
      check_deepmd_gen.append(1)

  if ( len(check_deepmd_gen) != 0 and all(i == 0 for i in check_deepmd_gen) ):
    str_print = 'Success: generate deepmd-kit tasks in %s' %(train_dir)
    str_print = data_op.str_wrap(str_print, 80, '  ')
    print (str_print, flush=True)
  else:
    log_info.log_error('Generating deepmd-kit tasks error, please check iteration %d' %(iter_id))
    exit()

def check_deepmd_run(train_dir, model_num):

  '''
  check_deepmd_run: check the status of dp jobs

  Args:
    train_dir: string
      train_dir is the directory of dp train.
    model_num: int
      model_num is the number of dp train models.
  Returns:
    None
  '''

  check_deepmd_run = []
  for i in range(model_num):
    ff_file_name_abs = ''.join((train_dir, '/', str(i), '/frozen_model.pb'))
    if ( os.path.exists(ff_file_name_abs) and os.path.getsize(ff_file_name_abs) ):
      check_deepmd_run.append(0)
    else:
      check_deepmd_run.append(1)

  if ( len(check_deepmd_run) != 0 and all(i == 0 for i in check_deepmd_run) ):
    print ('  Success: train %d models by deepmd-kit' %(model_num), flush=True)
  else:
    log_info.log_error('deepmd-kit running error, please check iteration %s' %(train_dir))
    exit()

def deepmd_parallel_cycle(work_dir, deepmd_train_dir, dp_path, cuda_dir, dp_cmd, model_str, \
                          device_choose_str, id_list_str, dp_job_per_node, parallel_exe, analyze_gpu):

  '''
  deepmd_parallel_cycle: run deepmd training in parallel

  Args:
    work_dir: string
      work_dir is the working directory of CP2K_kit.
    deepmd_train_dir: string
      deepmd_train_dir is the directory of deepmd-kit training.
    dp_path: string
      dp_path is the path of deepmd-kit.
    cuda_dir: string
      cuda_dir is the directory of cuda.
    dp_cmd: string
      dp_cmd is the deepmd-kit command.
    model_str: string
      model_str is the string containing model index.
    device_choose_str: string
      device_choose_str is the string containing gpu devices.
    dp_job_per_node: int
      dp_job_per_node is the number of deepmd-kit jobs in one node.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    ssh: bool
      ssh is whether to ssh to computational node.
  Returns:
    none
  '''

  run = '''
#! /bin/bash

model="%s"
device_id="%s"
id_list="%s"
direc=%s
parallel_exe=%s

model_arr=(${model///})
id_list_arr=(${id_list///})

num=${#model_arr[*]}

for ((i=0;i<=num-1;i++));
do
model_device_arr[i]="${model_arr[i]} ${id_list_arr[i]} $device_id"
done

for i in "${model_device_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d $direc/produce.sh {} $direc
''' %(model_str, device_choose_str, id_list_str, deepmd_train_dir, parallel_exe, dp_job_per_node)

  produce_1 = '''
#! /bin/bash

dp_path=%s
export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH

CUDA_DIR=%s
export PATH=$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH

export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

x=$1
direc=$2

x_arr=(${x///})

((id=${x_arr[1]}+2))
device_id=${x_arr[id]}

cd $direc/${x_arr[0]}
''' %(dp_path, cuda_dir)
  if analyze_gpu:
    produce_2 = '''
CUDA_VISIBLE_DEVICES=$device_id %s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
cd %s
''' %(dp_cmd, work_dir)
  else:
    produce_2 = '''
%s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
cd %s
''' %(dp_cmd, work_dir)

  run_file_name_abs = ''.join((deepmd_train_dir, '/run.sh'))
  with open(run_file_name_abs, 'w') as f:
    f.write(run)

  produce_file_name_abs = ''.join((deepmd_train_dir, '/produce.sh'))
  with open(produce_file_name_abs, 'w') as f:
    f.write(produce_1+produce_2)

  subprocess.run('chmod +x run.sh', cwd=deepmd_train_dir, shell=True)
  subprocess.run('chmod +x produce.sh', cwd=deepmd_train_dir, shell=True)
  try:
    subprocess.run("bash -c './run.sh'", cwd=deepmd_train_dir, shell=True)
  except subprocess.CalledProcessError as err:
    log_info.log_error('Running error: %s command running error in %s' %(err.cmd, deepmd_train_dir))
    exit()
 
def deepmd_parallel(work_dir, iter_id, use_prev_model, start, end, parallel_exe, \
                    dp_path, host, ssh, device, cuda_dir, dp_version, analyze_gpu, base):

  '''
  deepmd_parallel: run deepmd calculation in parallel.

  Args:
    work_dir: string
      work_dir is the working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    use_prev_model: bool
      use_prev_model is whether we need to use previous model.
    start: int
      start is the starting model id.
    end: int
      end is the endding model id.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    dp_path: string
      dp_path is the path of deepmd-kit.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether to ssh to computational node.
    device: 2-d string list
      device is the GPU device.
    cuda_dir: string
      cuda_dir is the directory of cuda.
  Returns:
    none
  '''

  model_num = end-start+1
  dp_cmd = pre_process(work_dir, iter_id, use_prev_model, start, end, dp_version, base)
  deepmd_train_dir = ''.join((work_dir, '/iter_', str(iter_id), '/01.train'))

  #Case 1
  if ( len(host) == 1 and len(device[0]) == 0 ):
    run = '''
#! /bin/bash

direc=%s
parallel_num=%d
run_start=%d
run_end=%d
parallel_exe=%s

seq $run_start $run_end | $parallel_exe -j $parallel_num $direc/produce.sh {} $direc
''' %(deepmd_train_dir, model_num, start, end, parallel_exe)

    produce = '''
#! /bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

dp_path=%s

export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH

x=$1
direc=$2
cd $direc/$x
%s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
cd %s
''' %(dp_path, dp_cmd, work_dir)

    run_file_name_abs = ''.join((deepmd_train_dir, '/run.sh'))
    with open(run_file_name_abs, 'w') as f:
      f.write(run)

    produce_file_name_abs = ''.join((deepmd_train_dir, '/produce.sh'))
    with open(produce_file_name_abs, 'w') as f:
      f.write(produce)

    subprocess.run('chmod +x run.sh', cwd=deepmd_train_dir, shell=True)
    subprocess.run('chmod +x produce.sh', cwd=deepmd_train_dir, shell=True)
    try:
      subprocess.run("bash -c './run.sh'", cwd=deepmd_train_dir, shell=True)
    except subprocess.CalledProcessError as err:
      log_info.log_error('Running error: %s command running error in %s' %(err.cmd, deepmd_train_dir))

  #Case 2
  if ( len(host) > 1 and all(len(i) == 0 for i in device) ):
    host_list = []
    for i in range(len(host)):
      host_list.append('-S' + ' ' + host[i])
    if ( len(host) < model_num ):
      host_list = host_list*math.ceil(model_num/len(host))
    host_list = host_list[0:model_num]
    host_comb = data_op.comb_list_2_str(host_list, ' ')

    run = '''
#! /bin/bash

direc=%s
parallel_num=%d
run_start=%d
run_end=%d
parallel_exe=%s

seq $run_start $run_end | $parallel_exe -j $parallel_num %s $direc/produce.sh {} $direc
''' %(deepmd_train_dir, math.ceil(model_num/len(host)), start, end, parallel_exe, host_comb)

    produce = '''
#! /bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

dp_path=%s

export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH

x=$1
direc=$2
cd $direc/$x
%s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
cd %s
''' %(dp_path, dp_cmd, work_dir)

    run_file_name_abs = ''.join((deepmd_train_dir, '/run.sh'))
    with open(run_file_name_abs, 'w') as f:
      f.write(run)

    produce_file_name_abs = ''.join((deepmd_train_dir, '/produce.sh'))
    with open(produce_file_name_abs, 'w') as f:
      f.write(produce)

    subprocess.run('chmod +x run.sh', cwd=deepmd_train_dir, shell=True)
    subprocess.run('chmod +x produce.sh', cwd=deepmd_train_dir, shell=True)
    try:
      subprocess.run("bash -c './run.sh'", cwd=deepmd_train_dir, shell=True)
    except subprocess.CalledProcessError as err:
      log_info.log_error('Running error: %s command running error in %s' %(err.cmd, deepmd_train_dir))

  #Case 3
  #If there is just 1 gpu device in the node, we prefer to run deepmd in serial.
  if ( len(host) == 1 and len(device[0]) == 1 ):
    run = '''
#! /bin/bash

dp_path=%s

export PATH=$dp_path/bin:$PATH
export LD_LIBRARY_PATH=$dp_path/lib:$LD_LIBRARY_PATH

CUDA_DIR=%s
export PATH=$CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH

export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

%s
dp freeze -o frozen_model.pb 1>> log.err 2>> log.err
''' %(dp_path, cuda_dir, dp_cmd)
    for i in range(model_num):
      deepmd_train_i_dir = ''.join((deepmd_train_dir, '/', str(i)))
      run_file_name_abs = ''.join((deepmd_train_i_dir, '/run.sh'))
      with open(run_file_name_abs, 'w') as f:
        f.write(run)
      subprocess.run('chmod +x run.sh', cwd=deepmd_train_i_dir, shell=True)
      try:
        subprocess.run("bash -c './run.sh'", cwd=deepmd_train_i_dir, shell=True)
      except subprocess.CalledProcessError as err:
        log_info.log_error('Running error: %s command running error in %s' %(err.cmd, deepmd_train_i_dir))

  #Case 4
  if ( len(device) == 1 and len(device[0]) > 1 ):
    device_num = len(device[0])
    dp_job_per_node = model_num
    if ( device_num < dp_job_per_node ):
      dp_job_per_node = device_num
    cycle = math.ceil(model_num/dp_job_per_node)

    run_start = 0
    run_end = run_start+dp_job_per_node-1
    if ( run_end > model_num-1 ):
      run_end=model_num-1

    for i in range(cycle):
      device_choose = device[0][0:(run_end-run_start+1)]
      id_list = data_op.gen_list(0, len(device_choose)-1, 1)
      id_list_str = data_op.comb_list_2_str(id_list, ' ')
      device_choose_str = data_op.comb_list_2_str(device_choose, ' ')
      model_list = data_op.gen_list(run_start, run_end, 1)
      model_str = data_op.comb_list_2_str(model_list, ' ')
      deepmd_parallel_cycle(work_dir, deepmd_train_dir, dp_path, cuda_dir, dp_cmd, model_str, \
                            device_choose_str, id_list_str, dp_job_per_node, parallel_exe, analyze_gpu)

      run_start = run_start + dp_job_per_node
      run_end = run_end + dp_job_per_node
      if ( run_end > model_num-1 ):
        run_end = model_num-1

def run_deepmd_ws(work_dir, iter_id, use_prev_model, parallel_exe, dp_path, host, \
                  ssh, device, cuda_dir, dp_version, analyze_gpu, base):

  '''
  run_deepmd_ws: kernel function to run deepmd for workstation mode.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    use_prev_model: bool
      use_prev_model is whether we need to use previous model.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    dp_path: string
      dp_path is the path of deepmd-kit.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether to ssh to computational node.
    device: 2-d string list
      device is the GPU device.
    cuda_dir: string
      cuda_dir is the directory of cuda.
    dp_job_per_node: int
      dp_job_per_node is the number of deepmd-kit jobs in one node.
    dp_version: string
      dp_version is the version of dp executable file.
    analyze_gpu: bool
      analyze_gpu is whether program need to analyze GPU.
  Returns:
    none
  '''

  import subprocess

  train_dir = ''.join((work_dir, '/iter_', str(iter_id), '/01.train'))
  model_num = len(call.call_returns_shell(train_dir, "ls -ll |awk '/^d/ {print $NF}'"))

  #Check generating deepmd tasks
  check_gen_deepmd(train_dir, model_num)

  if ( not all(len(i) == 0 for i in device) and cuda_dir == 'none' ):
    log_info.log_error('Input error: there are gpu devices in nodes, but cuda_dir is none, please set cuda directory in deepff/environ/cuda_dir')
    exit()

  #Run deepmd-kit tasks
  deepmd_parallel(work_dir, iter_id, use_prev_model, 0, model_num-1, parallel_exe, 
                  dp_path, host, ssh, device, cuda_dir, dp_version, analyze_gpu, base)

  #Check the deepmd tasks.
  check_deepmd_run(train_dir, model_num)

def run_deepmd_as(work_dir, iter_id, dp_queue, dp_core_num, dp_gpu_num, max_dp_job, submit_system, \
                  use_prev_model, dp_path, cuda_dir, dp_version, analyze_gpu, base):

  '''
  run_deepmd_as: kernel function to run deepmd for auto_submit mode.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    dp_queue: string
      dp_queue is the queue name of dp job.
    dp_core_num: int
      dp_core_num is the number of cores for each deepmd job.
    dp_gpu_num: int
      dp_gpu_num is the number of gpus for each deepmd job.
    max_dp_job: int
      max_dp_job is the maximum number of dp job.
    submit_system: string
      submit_system is the submition system.
    use_prev_model: bool
      use_prev_model is whether we need to use previous model.
    dp_path: string
      dp_path is the path of deepmd-kit.
    cuda_dir: string
      cuda_dir is the directory of cuda.
    dp_version: string
      dp_version is the version of dp executable file.
    analyze_gpu: bool
      analyze_gpu is whether program need to analyze GPU.
  Returns:
    none
  '''

  import time
  import subprocess

  if ( dp_gpu_num > 0 and cuda_dir == 'none' ):
    log_info.log_error('Input error: there are gpu devices in nodes, but cuda_dir is none, please set cuda directory in deepff/environ/cuda_dir')
    exit()

  train_dir = ''.join((work_dir, '/iter_', str(iter_id), '/01.train'))
  model_num = len(call.call_returns_shell(train_dir, "ls -ll |awk '/^d/ {print $NF}'"))

  #Check generating deepmd tasks
  check_gen_deepmd(train_dir, model_num)

  dp_queue = dp_queue*max_dp_job
  dp_queue_sub = dp_queue[0:max_dp_job]
  cycle = math.ceil(model_num/max_dp_job)
  for i in range(cycle):
    train_id_cycle = []
    rand_cycle = []
    for j in range(max_dp_job):
      train_id = i*max_dp_job+j
      if ( train_id < model_num ):
        train_id_cycle.append(train_id)
        dp_cmd = pre_process(work_dir, iter_id, use_prev_model, train_id, train_id, dp_version, base)
        train_id_dir = ''.join((train_dir, '/', str(train_id)))
        rand_int = np.random.randint(10000000000)
        rand_cycle.append(rand_int)
        job_label = ''.join(('dp_', str(rand_int))) 
        flag_file_name_abs = ''.join((train_id_dir, '/success.flag'))
        if ( os.path.exists(flag_file_name_abs) ):
          subprocess.run('rm %s' %(flag_file_name_abs), cwd=train_id_dir, shell=True)
        if ( submit_system == 'lsf' ):
          submit_file_name_abs = ''.join((train_id_dir, '/dp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( dp_gpu_num > 0 and not analyze_gpu ):
              script_1 = gen_shell_str.gen_lsf_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_lsf_gpu_set(1, dp_core_num)
              script_3 = gen_shell_str.gen_cd_lsfcwd()
              script_4 = gen_shell_str.gen_dp_env(dp_path)
              script_5 = gen_shell_str.gen_cuda_env(cuda_dir)
              script_6 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( dp_gpu_num > 0 and analyze_gpu ):
              script_1 = gen_shell_str.gen_lsf_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_lsf_gpu_set(1, dp_core_num)
              script_3 = gen_shell_str.gen_cd_lsfcwd()
              script_4 = gen_shell_str.gen_dp_env(dp_path)
              script_5 = gen_shell_str.gen_cuda_env(cuda_dir)
              script_6 = gen_shell_str.gen_gpu_analyze(1,1)
              script_7 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6+script_7)

            if ( dp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_lsf_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_cd_lsfcwd()
              script_3 = gen_shell_str.gen_dp_env(dp_path)
              script_4 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4)

          subprocess.run('bsub < ./dp.sub', cwd=train_id_dir, shell=True, stdout=subprocess.DEVNULL)

        if ( submit_system == 'pbs' ):
          submit_file_name_abs = ''.join((train_id_dir, '/dp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( dp_gpu_num > 0 ):
              script_1 = gen_shell_str.gen_pbs_normal(dp_queue_sub[j], dp_core_num, dp_gpu_num, job_label)
              script_2 = gen_shell_str.gen_cd_pbscwd()
              script_3 = gen_shell_str.gen_dp_env(dp_path)
              script_4 = gen_shell_str.gen_cuda_env(cuda_dir)
              script_5 = gen_shell_str.gen_gpu_analyze(1)
              script_6 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( dp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_pbs_normal(dp_queue_sub[j], dp_core_num, dp_gpu_num, job_label)
              script_2 = gen_shell_str.gen_cd_pbscwd()
              script_3 = gen_shell_str.gen_dp_env(dp_path)
              script_4 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4)

          subprocess.run('qsub ./dp.sub', cwd=train_id_dir, shell=True, stdout=subprocess.DEVNULL)

        if ( submit_system == 'slurm' ):
          submit_file_name_abs = ''.join((train_id_dir, '/dp.sub'))
          with open(submit_file_name_abs, 'w') as f:
            if ( dp_gpu_num > 0 and not analyze_gpu):
              script_1 = gen_shell_str.gen_slurm_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_slurm_gpu_set(1)
              script_3 = gen_shell_str.gen_dp_env(dp_path)
              script_4 = gen_shell_str.gen_cuda_env(cuda_dir)
              script_5 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4+script_5)

            if ( dp_gpu_num > 0 and analyze_gpu):
              script_1 = gen_shell_str.gen_slurm_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_slurm_gpu_set(1)
              script_3 = gen_shell_str.gen_dp_env(dp_path)
              script_4 = gen_shell_str.gen_cuda_env(cuda_dir)
              script_5 = gen_shell_str.gen_gpu_analyze(1)
              script_6 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3+script_4+script_5+script_6)

            if ( dp_gpu_num == 0 ):
              script_1 = gen_shell_str.gen_slurm_normal(dp_queue_sub[j], dp_core_num, job_label)
              script_2 = gen_shell_str.gen_dp_env(dp_path)
              script_3 = gen_shell_str.gen_dp_cmd(dp_cmd)
              f.write(script_1+script_2+script_3)

          subprocess.run('sbatch ./dp.sub', cwd=train_id_dir, shell=True, stdout=subprocess.DEVNULL)

    job_id = []
    failure_id = []
    for j in range(len(train_id_cycle)):
      job_id_j = process.get_job_id(work_dir, submit_system, 'dp_', rand_cycle[j])
      if ( job_id_j > 0 ):
        job_id.append(job_id_j)
      else:
        failure_id.append(train_id_cycle[j])
    if ( len(job_id) == len(train_id_cycle) ):
      for j in range(len(train_id_cycle)):
        str_print = 'Success: submit dp train job for model %d in iteration %d with job id %d' \
                     %(train_id_cycle[j], iter_id, job_id[j])
        str_print = data_op.str_wrap(str_print, 80, '  ')
        print (str_print, flush=True)
      while True:
        time.sleep(10)
        judge = []
        for j in train_id_cycle:
          flag_file_name = ''.join((train_dir, '/', str(j), '/success.flag'))
          judge.append(os.path.exists(flag_file_name))
        if all(judge):
          break
    else:
      log_info.log_error('Fail to submit dp train job for model %s in iteration %d' \
                         %(data_op.comb_list_2_str(failure_id, ' '), iter_id))
      exit()

  #Check the deepmd tasks.
  check_deepmd_run(train_dir, model_num)

if __name__ == '__main__':
  from DPFlow.deepff import deepmd_run
  from DPFlow.tools import read_input
  from DPFlow.deepff import load_data
  from DPFlow.deepff import check_deepff

  deepff_key = ['deepmd', 'lammps', 'cp2k', 'model_devi', 'environ']
  work_dir = '/home/lujunbo/code/github/DPFlow/deepff/work_dir'

  deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic = \
  read_input.dump_info(work_dir, 'input.inp', deepff_key)
  proc_num = 4
  deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic = \
  check_deepff.check_inp(deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic, proc_num)

  seed = [1,2,3,4]
  numb_test = int(deepmd_dic['training']['numb_test'])

  i = 0
  init_train_data = []
  cmd = "mkdir %s" % ('init_train_data')
  call.call_simple_shell(work_dir, cmd)
  train_dic = deepmd_dic['training']
  for key in train_dic:
    if ( 'system' in key):
      init_train_key_dir = train_dic[key]['directory']
      proj_name = train_dic[key]['proj_name']
      save_dir = ''.join((work_dir, '/init_train_data/data_', str(i)))
      init_train_data.append(save_dir)
      load_data.load_data_from_dir(init_train_key_dir, work_dir, save_dir, proj_name)
      load_data.raw_to_set(save_dir, 1)
      i = i+1

  #Test gen_deepmd_task function
  deepmd_run.gen_deepmd_task(deepmd_dic, work_dir, 0, init_train_data, seed, numb_test)
  #Test run_deepmd function
  deepmd_run.run_deepmd(work_dir, 0)
