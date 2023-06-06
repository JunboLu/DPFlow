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

def check_gen_lmpfrc(lmp_dir, sys_num, active_type, use_bias_tot):

  '''
  check_gen_lmpfrc: check the status of generating lammps force jobs

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps.
    sys_num: int
      sys_num is the number of lammps systems.
    active_type: string
      active_type is the type of active learning.
    use_bias_tot: bool
      use_bias_tot is whethet using metadynamics for whole systems.
  Returns:
    None
  '''

  check_lmp_frc_gen = []
  for i in range(sys_num):
    lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
    task_num = process.get_task_num(lmp_sys_dir)
    if ( active_type == 'model_devi' or use_bias_tot[i] ):
      for j in range(task_num):
        lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
        model_num = process.get_lmp_model_num(lmp_sys_task_dir)
        for k in range(model_num):
          model_dir = ''.join((lmp_sys_task_dir, '/model_', str(k)))
          traj_num = process.get_traj_num(model_dir)
          for l in range(traj_num):
            traj_dir = ''.join((model_dir, '/traj_', str(l)))
            if ( not use_bias_tot[i] and k == 0 ) :
              check_file_name_abs = ''.join((traj_dir, '/atom.dump'))
            else:
              check_file_name_abs = ''.join((traj_dir, '/frc_in.lammps'))
            if ( os.path.exists(check_file_name_abs) and os.path.getsize(check_file_name_abs) != 0 ):
              check_lmp_frc_gen.append(0)
            else:
              check_lmp_frc_gen.append(1)
    else:
      for j in range(task_num):
        check_lmp_frc_gen.append(0)

  if ( len(check_lmp_frc_gen) != 0 and all(i == 0 for i in check_lmp_frc_gen) ):
    str_print = 'Success: generating lammps force calculation file in %s' %(lmp_dir)
    str_print = data_op.str_wrap(str_print, 80, '  ')
    print (str_print, flush=True)
  else:
    log_info.log_error('Generating lammps force calculation tasks error, please check %s' %(lmp_dir))
    exit()

def check_lmpfrc_run(lmp_dir, sys_num, use_bias_tot, atoms_num_tot):

  '''
  check_lmpfrc_run: check the statu of lammps force calculation

  Args:
    lmp_dir: string
      lmp_dir is the directory of lammps calculation.
    sys_num: int
      sys_num is the number of systems.
    use_bias_tot: bool
      use_myd_tot is whethet using metadynamics for whole systems.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
  Returns:
    check_lmp_frc_run: 4-d int list
      check_lmp_frc_run is the statu of lammps force calculation.
  '''

  check_lmp_frc_run = []
  for i in range(sys_num):
    lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
    cmd = "ls | grep %s" % ('task_')
    task_num = process.get_task_num(lmp_sys_dir)
    check_lmp_frc_run_i = []
    for j in range(task_num):
      lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
      model_num = process.get_lmp_model_num(lmp_sys_task_dir)
      check_lmp_frc_run_ij = []
      for k in range(model_num):
        model_dir = ''.join((lmp_sys_task_dir, '/model_', str(k)))
        traj_num = process.get_traj_num(model_dir)
        check_lmp_frc_run_ijk = []
        for l in range(traj_num):
          traj_dir = ''.join((model_dir, '/traj_', str(l)))
          dump_file_name_abs = ''.join((traj_dir, '/atom.dump'))
          log_file_name_abs = ''.join((traj_dir, '/lammps.out'))
          if ( os.path.exists(dump_file_name_abs) ):
            if ( not file_tools.is_binary(dump_file_name_abs) and \
                 len(open(dump_file_name_abs, 'r').readlines()) == atoms_num_tot[i]+9 ):
              if ( use_bias_tot[i] or k != 0 ):
                if ( os.path.exists(log_file_name_abs) and \
                     file_tools.grep_line_num('Step', log_file_name_abs, traj_dir) != 0 and \
                     file_tools.grep_line_num('Loop time', log_file_name_abs, traj_dir) != 0 ):
                  check_lmp_frc_run_ijk.append(0)
                else:
                  check_lmp_frc_run_ijk.append(1)
              else:
                check_lmp_frc_run_ijk.append(0)
            else:
              check_lmp_frc_run_ijk.append(1)
          else:
            check_lmp_frc_run_ijk.append(1)
        check_lmp_frc_run_ij.append(check_lmp_frc_run_ijk)
      check_lmp_frc_run_i.append(check_lmp_frc_run_ij)
    check_lmp_frc_run.append(check_lmp_frc_run_i)

  return check_lmp_frc_run

def lmpfrc_parallel(model_dir, work_dir, task_index, parallel_exe, lmp_path, \
                    lmp_exe, lmp_md_job_per_node, host_name, ssh):

  '''
  lmpfrc_parallel: run lammps force calculation in parallel.

  Args:
    model_dir: string
      model_dir is directory for any model.
    work_dir: string
      work_dir is the working directory of DPFlow.
    task_index: 1-d int list
      task_index is the index of tasks.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    lmp_path: string
      lmp_path is the path of lammps.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_md_job_per_node: int
      lmp_md_job_per_node is the number lammps job in each node.
    host_name: string
      host_name is the string host name.
    ssh: bool
      ssh is whether we need to ssh.
  Returns:
    none
  '''

  #run lammps in 1 thread. Here we just run force, it is a single
  #point calculation.

  task_index_str = data_op.comb_list_2_str(task_index, ' ')
  run_1 = '''
#! /bin/bash

direc=%s
task_index="%s"
parallel_exe=%s

ulimit -u 65535

task_index_arr=(${task_index///})
num=${#task_index_arr[*]}
''' %(model_dir, task_index_str, parallel_exe)

  if ssh:
    run_2 ='''
for i in "${task_index_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --controlmaster -S %s --sshdelay 0.2  $direc/produce.sh {} $direc
''' %(lmp_md_job_per_node, host_name)
  else:
    run_2 ='''
for i in "${task_index_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --delay 0.2 $direc/produce.sh {} $direc
''' %(lmp_md_job_per_node)

  produce = '''
#! /bin/bash

lmp_path=%s

export PATH=$lmp_path/bin:$PATH
export LD_LIBRARY_PATH=$lmp_path/lib:$LD_LIBRARY_PATH

x=$1
direc=$2
new_direc=$direc/traj_$x
cd $new_direc
%s < $new_direc/frc_in.lammps 1> $new_direc/lammps.out 2> $new_direc/lammps.err
cd %s
''' %(lmp_path, lmp_exe, work_dir)

  produce_file_name_abs = ''.join((model_dir, '/produce.sh'))
  with open(produce_file_name_abs, 'w') as f:
    f.write(produce)
  run_file_name_abs = ''.join((model_dir, '/run.sh'))
  with open(run_file_name_abs, 'w') as f:
    f.write(run_1+run_2)

  subprocess.run('chmod +x produce.sh', cwd=model_dir, shell=True)
  subprocess.run('chmod +x run.sh', cwd=model_dir, shell=True)
  try:
    subprocess.run("bash -c './run.sh'", cwd=model_dir, shell=True)
  except subprocess.CalledProcessError as err:
    log_info.log_error('Running error: %s command running error in %s' %(err.cmd, model_dir))

def run_undo_lmpfrc(work_dir, iter_id, use_bias_tot, atoms_num_tot, active_type, mode, \
                    parallel_exe, lmp_path, lmp_exe, lmp_frc_job_per_node, host_info, \
                    ssh, lmp_queue, lmp_core_num, lmp_gpu_num, submit_system, analyze_gpu):

  '''
  rum_undo_lmpfrc: run uncompleted lammps force calculation.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    use_bias_tot: bool
      use_bias_tot is whethet using metadynamics for whole systems.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
    active_type: string
      active_type is the type of active learning.
    mode: string
      mode is the runing mode of DPFlow.
    parallel_exe: string
      parallel_exe is parallel exacutable file.
    lmp_path: string
      lmp_path is the path of lammps.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_frc_job_per_node: int
      lmp_frc_job_per_node is the number of lammps force jobs.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether we need to ssh.
    lmp_queue: string
      lmp_queue is the queue name of lammps job.
    lmp_core_num: int
      lmp_core_num is the number of cores for each lammps md job.
    lmp_gpu_num: int
      lmp_gpu_num is the number of gpus for each lammps md job.
    submit_system: string
      submit_system is the submition system.
  Returns :
    none
  '''

  lmp_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))

  cmd = 'ls | grep %s' % ('sys_')
  sys_num = len(call.call_returns_shell(lmp_dir, cmd))

  for cycle_run in range(100):
    check_lmp_frc_run = check_lmpfrc_run(lmp_dir, sys_num, use_bias_tot, atoms_num_tot)
    lmp_frc_statu = []
    for i in range(sys_num):
      lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
      task_num = process.get_task_num(lmp_sys_dir)
      if ( active_type == 'model_devi' or use_bias_tot[i] ):
        for j in range(task_num):
          lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
          model_num = process.get_lmp_model_num(lmp_sys_task_dir)
          for k in range(model_num):
            if ( len(check_lmp_frc_run[i][j][k]) != 0 and all(m == 0 for m in check_lmp_frc_run[i][j][k]) ):
              lmp_frc_statu.append(0)
            else:
              lmp_frc_statu.append(1)
              undo_task = [index for (index,value) in enumerate(check_lmp_frc_run[i][j][k]) if value==1]
              model_dir = ''.join((lmp_dir, '/sys_', str(i), '/task_', str(j), '/model_', str(k)))
              traj_num = process.get_traj_num(model_dir)
              if ( len(undo_task) != 0 ):
                if ( mode == 'workstation' ):
                  cycle = math.ceil(len(undo_task)/(lmp_frc_job_per_node*len(host)))
                  start = 0
                  end = min(len(undo_task), lmp_frc_job_per_node*len(host))
                  for l in range(cycle):
                    task_index = undo_task[start:end]
                    lmpfrc_parallel(model_dir, work_dir, task_index, parallel_exe, lmp_path, \
                                    lmp_exe, lmp_frc_job_per_node, host_info, ssh)
                    start = min(len(undo_task)-1, start+lmp_frc_job_per_node*len(host))
                    end = min(len(undo_task), end+lmp_frc_job_per_node*len(host))
                elif ( mode == 'auto_submit' ):
                  for undo_id in undo_task:
                    undo_task_flag_file = ''.join((model_dir, '/traj_', str(undo_id), '/success.flag'))
                    if ( os.path.exists(undo_task_flag_file) ):
                      subprocess.run('rm %s' %(undo_task_flag_file), \
                                     cwd=''.join((model_dir, '/traj_', str(undo_id))), shell=True)
                  submit_lmpfrc_discrete(model_dir, iter_id, undo_task, lmp_queue[0], lmp_exe, lmp_core_num, \
                                         lmp_gpu_num, parallel_exe, submit_system, analyze_gpu)
                  while True:
                    time.sleep(10)
                    judge = []
                    for l in range(traj_num):
                      flag_file_name = ''.join((model_dir, '/traj_', str(l), '/success.flag'))
                      judge.append(os.path.exists(flag_file_name))
                    if all(judge):
                       break
      else:
        for j in range(task_num):
          lmp_frc_statu.append(0)
    if ( len(lmp_frc_statu) !=0 and all(i == 0 for i in lmp_frc_statu) ):
      break
    if ( cycle_run == 99 and not all(i == 0 for i in lmp_frc_statu) ):
      lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
      task_num = process.get_task_num(lmp_sys_dir)
      for j in range(task_num):
        for k in range(model_num):
          undo_task = [index for (index,value) in enumerate(check_lmp_frc_run[i][j][k]) if value==1]
          if ( len(undo_task) != 0 ):
            undo_task_str = data_op.comb_list_2_str(undo_task, ' ')
            str_print = '  Warning: force calculations fail for tasks %s in system %d in task %d by model %d' \
                        %(undo_task_str, i, j, k)
            str_print = data_op.str_wrap(str_print, 80, '  ')
            print (str_print, flush=True)
      exit()

def run_lmpfrc_ws(work_dir, iter_id, lmp_path, lmp_exe, parallel_exe, lmp_frc_job_per_node, \
                  host, device, atoms_num_tot, use_bias_tot, active_type):

  '''
  rum_lmpfrc_ws: kernel function to run lammps force calculation.

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
      parallel_exe is parallel exacutable file.
    lmp_frc_job_per_node: int
      lmp_frc_job_per_node is the number of lammps force jobs.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether we need to ssh.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
    use_bias_tot: bool
      use_bias_tot is whethet using metadynamics for whole systems.
    active_type: string
      active_type is the type of active learning.
  Returns :
    none
  '''

  lmp_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))

  cmd = 'ls | grep %s' % ('sys_')
  sys_num = len(call.call_returns_shell(lmp_dir, cmd))

  #Check generating lammps force tasks.
  check_gen_lmpfrc(lmp_dir, sys_num, active_type, use_bias_tot)

  if ( len(host) > 1 and all(len(i) == 0 for i in device) ):
    ssh = True
  else:
    ssh = False

  #Run lammps force.
  for i in range(sys_num):
    if ( active_type == 'model_devi' or use_bias_tot[i] ):
      lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
      task_num = process.get_task_num(lmp_sys_dir)
      for j in range(task_num):
        lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
        model_num = process.get_lmp_model_num(lmp_sys_task_dir)
        for k in range(model_num):
          model_dir = ''.join((lmp_sys_task_dir, '/model_', str(k)))
          host_name_proc = []
          for l in range(len(host)):
            host_name_proc.append(''.join((str(lmp_frc_job_per_node), '/', host[l])))
          host_info = data_op.comb_list_2_str(host_name_proc, ',')
          traj_num = process.get_traj_num(model_dir)

          calculated_id = 0
          undo_task = []
          for l in range(traj_num):
            traj_dir = ''.join((model_dir, '/traj_', str(l)))
            dump_file_name_abs = ''.join((traj_dir, '/atom.dump'))
            log_file_name_abs = ''.join((traj_dir, '/lammps.out'))
            if ( os.path.exists(dump_file_name_abs) ):
              if ( not file_tools.is_binary(dump_file_name_abs) and \
                 len(open(dump_file_name_abs, 'r').readlines()) == atoms_num_tot[i]+9 and \
                 os.path.exists(log_file_name_abs) and \
                 file_tools.grep_line_num('Step', log_file_name_abs, traj_dir) != 0 and \
                 file_tools.grep_line_num('Loop time', log_file_name_abs, traj_dir) != 0 ):
                pass
              else:
                undo_task.append(l)
            else:
              undo_task.append(l)

          start = 0
          end = min(len(undo_task), start+lmp_frc_job_per_node*len(host))
          cycle = math.ceil(len(undo_task)/(lmp_frc_job_per_node*len(host)))

          for l in range(cycle):
            if ( use_bias_tot[i] or k != 0 ):
              task_index = undo_task[start:end]
              lmpfrc_parallel(model_dir, work_dir, task_index, parallel_exe, lmp_path, \
                              lmp_exe, lmp_frc_job_per_node, host_info, ssh)
            start = min(len(undo_task)-1, start+lmp_frc_job_per_node*len(host))
            end = min(len(undo_task), end+lmp_frc_job_per_node*len(host))

  run_undo_lmpfrc(work_dir, iter_id, use_bias_tot, atoms_num_tot, active_type, 'workstation', \
                  parallel_exe, lmp_path, lmp_exe, lmp_frc_job_per_node, host_info, \
                  ssh, None, None, None, None, None)

  print ('  Success: lammps force calculations for %d systems by lammps' %(sys_num), flush=True)

def submit_lmpfrc_discrete(model_dir, iter_id, undo_task, lmp_queue, lmp_exe, lmp_core_num, \
                           lmp_gpu_num, parallel_exe, submit_system, analyze_gpu):

  '''
  submit_lmpfrc_discrete: submit discrete lammps force jobs to remote host.

  Args:
    model_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    undo_task: 1-d int list
      undo_task are id of uncompleted lammps force tasks.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_queue: string
      lmp_queue is the queue name of lammps job.
    lmp_core_num: int
      lmp_core_num is the number of cores for each lammps md job.
    lmp_gpu_num: int
      lmp_gpu_num is the number of gpus for each lammps md job.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    submit_system: string
      submit_system is the submition system.
  Returns:
    none
  '''

  task_index = data_op.comb_list_2_str(undo_task, ' ')

  if ( submit_system == 'lsf' ):

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 and not analyze_gpu ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_cd_lsfcwd()
      script_4 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num > 0 and analyze_gpu ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_cd_lsfcwd()
      script_4 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_5 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4+script_5)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_lsfcwd()
      script_3 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3)

    subprocess.run('bsub < ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'pbs' ):

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 ):
      script_1 = gen_shell_str.gen_pbs_normal(lmp_queue, lmp_core_num, lmp_gpu_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_pbscwd()
      script_3 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_4 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_pbs_normal(lmp_queue, lmp_core_num, lmp_gpu_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_pbscwd()
      script_3 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3)

    subprocess.run('qsub ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'slurm' ):

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 ):
      script_1 = gen_shell_str.gen_slurm_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_slurm_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_4 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_slurm_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lmp_frc_dis(model_dir, task_index, parallel_exe, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2)

    subprocess.run('sbatch ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

def submit_lmpfrc_serial(model_dir, iter_id, start, end, lmp_queue, lmp_exe, lmp_core_num, \
                         lmp_gpu_num, parallel_exe, submit_system, analyze_gpu):

  '''
  submit_lmpfrc_serial: submit serial lammps force jobs to remote host.

  Args:
    model_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    start: int
      start is the staring task id.
    end: int
      end is the endding task id.
    lmp_queue: string
      lmp_queue is the queue name of lammps job.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_core_num: int
      lmp_core_num is the number of cores for each lammps md job.
    lmp_gpu_num: int
      lmp_gpu_num is the number of gpus for each lammps md job.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    submit_system: string
      submit_system is the submition system.
  Returns:
    none
  '''

  if ( submit_system == 'lsf' ): 

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 and not analyze_gpu ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_cd_lsfcwd()
      script_4 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num > 0 and analyze_gpu ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lsf_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_cd_lsfcwd()
      script_4 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_5 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4+script_5)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_lsf_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_lsfcwd()
      script_3 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3)

    subprocess.run('bsub < ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'pbs' ):

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 ):
      script_1 = gen_shell_str.gen_pbs_normal(lmp_queue, lmp_core_num, lmp_gpu_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_pbscwd()
      script_3 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_4 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_pbs_normal(lmp_queue, lmp_core_num, lmp_gpu_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_cd_pbscwd()
      script_3 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3)

    subprocess.run('qsub ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'slurm' ):

    submit_file_name_abs = ''.join((model_dir, '/lmp_frc.sub'))
    if ( lmp_gpu_num > 0 ):
      script_1 = gen_shell_str.gen_slurm_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_slurm_gpu_set(lmp_gpu_num)
      script_3 = gen_shell_str.gen_gpu_analyze(lmp_gpu_num)
      script_4 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2+script_3+script_4)

    elif ( lmp_gpu_num == 0 ):
      script_1 = gen_shell_str.gen_slurm_normal(lmp_queue, lmp_core_num, iter_id, 'lmp_frc')
      script_2 = gen_shell_str.gen_lmq_frc_ser(model_dir, parallel_exe, start, end, lmp_core_num)
      with open(submit_file_name_abs, 'w') as f:
        f.write(script_1+script_2)

    subprocess.run('sbatch ./lmp_frc.sub', cwd=model_dir, shell=True, stdout=subprocess.DEVNULL)

def run_lmpfrc_as(work_dir, iter_id, active_type, use_bias_tot, lmp_path, lmp_exe, lmp_queue, \
                  lmp_core_num, lmp_gpu_num, parallel_exe, submit_system, atoms_num_tot, analyze_gpu):

  '''
  run_lmpmdfrc_as: kernel function to run lammps force for auto_submit mode.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    active_type: string
      active_type is the type of active learning.
    use_bias_tot: bool
      use_bias_tot is whethet using metadynamics for whole systems.
    lmp_path: string
      lmp_path is the path of lammps.
    lmp_exe: string
      lmp_exe is the lammps executable file.
    lmp_queue: string
      lmp_queue is the queue name of lammps job.
    lmp_core_num: int
      lmp_core_num is the number of cores for each lammps md job.
    lmp_gpu_num: int
      lmp_gpu_num is the number of gpus for each lammps md job.
    parallel_exe: string
      parallel_exe is the parallel exacutable file.
    submit_system: string
      submit_system is the submition system.
  Returns:
    none
  '''

  lmp_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))

  cmd = 'ls | grep %s' % ('sys_')
  sys_num = len(call.call_returns_shell(lmp_dir, cmd))

  #Check generating lammps force tasks.
  check_gen_lmpfrc(lmp_dir, sys_num, active_type, use_bias_tot)

  #Run lammps force.
  for i in range(sys_num):
    if ( active_type == 'model_devi' or use_bias_tot[i] ):
      lmp_sys_dir = ''.join((lmp_dir, '/sys_', str(i)))
      task_num = process.get_task_num(lmp_sys_dir)
      for j in range(task_num):
        lmp_sys_task_dir = ''.join((lmp_sys_dir, '/task_', str(j)))
        model_num = process.get_lmp_model_num(lmp_sys_task_dir)
        for k in range(model_num):
          model_dir = ''.join((lmp_sys_task_dir, '/model_', str(k)))
          produce = '''
#! /bin/bash

lmp_path=%s

export PATH=$lmp_path/bin:$PATH
export LD_LIBRARY_PATH=$lmp_path/lib:$LD_LIBRARY_PATH

x=$1
direc=$2
new_direc=$direc/traj_$x
cd $new_direc
if [ -f "success.flag" ]; then
rm success.flag
fi
%s < $new_direc/frc_in.lammps 1> $new_direc/lammps.out 2> $new_direc/lammps.err
echo 'success' > success.flag
''' %(lmp_path, lmp_exe)

          produce_file_name_abs = ''.join((model_dir, '/produce.sh'))
          with open(produce_file_name_abs, 'w') as f:
            f.write(produce)
          subprocess.run('chmod +x produce.sh', cwd=model_dir, shell=True)
 
          traj_num = process.get_traj_num(model_dir)

          undo_task = []
          for l in range(traj_num):
            traj_dir = ''.join((model_dir, '/traj_', str(l)))
            dump_file_name_abs = ''.join((traj_dir, '/atom.dump'))
            log_file_name_abs = ''.join((traj_dir, '/lammps.out'))
            if ( os.path.exists(dump_file_name_abs) ):
              if ( not file_tools.is_binary(dump_file_name_abs) and \
                 len(open(dump_file_name_abs, 'r').readlines()) == atoms_num_tot[i]+9 and \
                 os.path.exists(log_file_name_abs) and \
                 file_tools.grep_line_num('Step', log_file_name_abs, traj_dir) != 0 and \
                 file_tools.grep_line_num('Loop time', log_file_name_abs, traj_dir) != 0 ):
                pass
              else:
                undo_task.append(l)
            else:
              undo_task.append(l)
          if ( len(undo_task) != 0 ):
            for undo_id in undo_task:
              undo_task_flag_file = ''.join((model_dir, '/traj_', str(undo_id), '/success.flag'))
              if ( os.path.exists(undo_task_flag_file) ):
                subprocess.run('rm %s' %(undo_task_flag_file), \
                               cwd=''.join((model_dir, '/traj_', str(undo_id))), shell=True)
          if ( len(undo_task) != 0 and len(undo_task) < 20 ):
            submit_lmpfrc_discrete(model_dir, iter_id, undo_task, lmp_queue[0], lmp_exe, lmp_core_num, \
                                   lmp_gpu_num, parallel_exe, submit_system, analyze_gpu)
          elif ( len(undo_task) > 20 ):
            submit_lmpfrc_serial(model_dir, iter_id, undo_task[0], undo_task[len(undo_task)-1], lmp_queue[0], \
                                 lmp_exe, lmp_core_num, lmp_gpu_num, parallel_exe, submit_system, analyze_gpu)
          while True:
            time.sleep(10)
            judge = []
            for l in range(traj_num):
              flag_file_name = ''.join((model_dir, '/traj_', str(l), '/success.flag'))
              judge.append(os.path.exists(flag_file_name))
            if all(judge):
              break

  run_undo_lmpfrc(work_dir, iter_id, use_bias_tot, atoms_num_tot, active_type, \
                  'auto_submit', parallel_exe, lmp_path, lmp_exe, None, None, None, \
                  lmp_queue, lmp_core_num, lmp_gpu_num, submit_system, analyze_gpu)

  print ('  Success: lammps force calculations for %d systems by lammps' %(sys_num), flush=True)

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
