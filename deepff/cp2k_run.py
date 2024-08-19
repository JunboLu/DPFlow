#! /usr/env/bin python

import os
import math
import time
import subprocess
from DPFlow.deepff import process
from DPFlow.deepff import gen_shell_str
from DPFlow.tools import *

def check_cp2k_gen(cp2k_calc_dir, sys_num):

  '''
  check_cp2k_gen: check the statu of generation of cp2k force jobs.

  Args:
    cp2k_calc_dir: string
      cp2k_calc_dir is the directory of cp2k calculation.
    sys_num: int
      sys_num is the number of systems.
  Returns:
    None
  '''

  check_cp2k_gen = []
  for i in range(sys_num):
    cp2k_sys_dir = ''.join((cp2k_calc_dir, '/sys_', str(i)))
    task_num, task_dir = process.get_task_num(cp2k_sys_dir, True)
    for j in range(task_num):
      cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/', task_dir[j]))
      traj_num = process.get_traj_num(cp2k_sys_task_dir)
      for k in range(traj_num):
        cp2k_sys_task_traj_dir = ''.join((cp2k_sys_task_dir, '/traj_', str(k)))
        inp_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/input.inp'))
        if ( os.path.exists(inp_file_name_abs) and os.path.getsize(inp_file_name_abs) != 0 ):
          check_cp2k_gen.append(0)
        else:
          check_cp2k_gen.append(1)
  if ( all(i == 0 for i in check_cp2k_gen) ):
    str_print = 'Success: generate cp2k tasks in %s' %(cp2k_calc_dir)
    str_print = data_op.str_wrap(str_print, 80, '  ')
    print (str_print, flush=True)
  else:
    log_info.log_error('Generating cp2k tasks error, please check %s' %(cp2k_calc_dir))
    exit()

def check_cp2k_job(cp2k_calc_dir, sys_num, atoms_num_tot):

  '''
  check_cp2k_job : check the statu of cp2k jobs

  Args:
    cp2k_calc_dir : string
      cp2k_calc_dir is the directory of cp2k calculation.
    sys_num : int
      sys_num is the number of systems.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
  Returns:
    check_cp2k_run : 3-d int list
      check_cp2k_run is the status of cp2k jobs
  '''

  check_cp2k_run = []
  for i in range(sys_num):
    cp2k_sys_dir = ''.join((cp2k_calc_dir, '/sys_', str(i)))
    task_num = process.get_task_num(cp2k_sys_dir)
    check_cp2k_run_i = []
    for j in range(task_num):
      check_cp2k_run_ij = []
      cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/task_', str(j)))
      traj_num = process.get_traj_num(cp2k_sys_task_dir)
      for k in range(traj_num):
        cp2k_sys_task_traj_dir = ''.join((cp2k_sys_task_dir, '/traj_', str(k)))
        frc_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k-1_0.xyz'))
        log_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k.out'))
        coord_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k-1.coordLog'))
        if ( os.path.exists(frc_file_name_abs) and os.path.exists(log_file_name_abs) and \
             os.path.exists(coord_file_name_abs) ):
          if ( not file_tools.is_binary(frc_file_name_abs) and \
               not file_tools.is_binary(log_file_name_abs) and \
               not file_tools.is_binary(coord_file_name_abs) and \
               len(open(frc_file_name_abs, 'r').readlines()) > atoms_num_tot[i] and \
               len(open(coord_file_name_abs, 'r').readlines()) > atoms_num_tot[i] and \
               file_tools.grep_line_num('Total energy:', log_file_name_abs, cp2k_sys_task_traj_dir) != 0 ):
            check_cp2k_run_ij.append(0)
          else:
            check_cp2k_run_ij.append(1)
        else:
          check_cp2k_run_ij.append(1)
      check_cp2k_run_i.append(check_cp2k_run_ij)
    check_cp2k_run.append(check_cp2k_run_i)

  return check_cp2k_run

def find_undo_task(cp2k_sys_task_dir, atoms_num, job_mode):

  '''
  find_undo_task: find the uncompleted cp2k force jobs.

  Args:
    cp2k_sys_task_dir: string
      cp2k_sys_task_dir is the directory of cp2k for one system and one task.
    atoms_num: int
      atoms_num is the number of atoms.
  Returns:
    undo_task: 1-d int list
      undo_task is the id of uncompleted cp2k force jobs.
  '''

  traj_num = process.get_traj_num(cp2k_sys_task_dir)
  undo_task = []
  for i in range(traj_num):
    cp2k_sys_task_traj_dir = ''.join((cp2k_sys_task_dir, '/traj_', str(i)))

    frc_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k-1_0.xyz'))
    log_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k.out'))
    coord_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/cp2k-1.coordLog'))
    flag_file_name_abs = ''.join((cp2k_sys_task_traj_dir, '/success.flag'))
    if ( job_mode == 'workstation' ):
      if ( os.path.exists(frc_file_name_abs) and os.path.exists(log_file_name_abs) and \
           os.path.exists(coord_file_name_abs) ):
        if ( not file_tools.is_binary(frc_file_name_abs) and \
             not file_tools.is_binary(log_file_name_abs) and \
             not file_tools.is_binary(coord_file_name_abs) and \
             len(open(frc_file_name_abs, 'r').readlines()) == atoms_num+5 and \
             len(open(coord_file_name_abs, 'r').readlines()) == atoms_num+8 and \
             file_tools.grep_line_num('Total energy:', log_file_name_abs, cp2k_sys_task_traj_dir) != 0 ):
          pass
        else:
          undo_task.append(i)
      else:
        undo_task.append(i)

    if ( job_mode == 'auto_submit' ):
      if ( os.path.exists(frc_file_name_abs) and os.path.exists(log_file_name_abs) and \
           os.path.exists(coord_file_name_abs) and os.path.exists(flag_file_name_abs) ):
        if ( not file_tools.is_binary(frc_file_name_abs) and \
             not file_tools.is_binary(log_file_name_abs) and \
             not file_tools.is_binary(coord_file_name_abs) and \
             len(open(frc_file_name_abs, 'r').readlines()) > atoms_num and \
             len(open(coord_file_name_abs, 'r').readlines()) > atoms_num ):
          pass
        else:
          undo_task.append(i)
      else:
        undo_task.append(i)

  return undo_task

def run_undo_cp2kfrc(work_dir, iter_id, cp2k_env_file, cp2k_exe, atoms_num_tot, job_mode, \
                     cp2k_job_per_node, host, ssh, proc_num_per_node, parallel_exe, \
                     cp2k_queue, max_cp2k_job, cp2k_core_num, submit_system):

  '''
  rum_undo_lmpfrc: run uncompleted cp2k force calculation.

  Args:
    work_dir: string
      work_dir is the workding directory of DPFlow.
    iter_id: int
      iter_id is current iteration number.
    cp2k_env_file: string
      cp2k_env_file is the environment setting file of cp2k.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
    cp2k_job_per_node: int
      cp2k_job_per_node is the job number of cp2k in each node.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether we need to ssh.
    proc_num_per_node: 1-d int list
      proc_num_per_node is the numbers of processor in each node.
    cp2k_queue: string
      cp2k_queue is the queue name of cp2k job.
    max_cp2k_job: int
      max_cp2k_job is the maximum number of sumbition for cp2k jobs.
    cp2k_core_num: int
      cp2k_core_num is the number of cores for each cp2k job.
    submit_system: string
      submit_system is the submition system.
  Returns:
    none
  '''

  cp2k_calc_dir = ''.join((work_dir, '/iter_', str(iter_id), '/03.cp2k_calc'))
  sys_num = process.get_sys_num(cp2k_calc_dir)

  for cycle in range(10):
    check_cp2k_run = check_cp2k_job(cp2k_calc_dir, sys_num, atoms_num_tot)
    if ( all(i == 0 for i in data_op.list_reshape(data_op.list_reshape(check_cp2k_run))) ):
      print ('  Success: ab initio force calculations for %d systems by cp2k' %(sys_num), flush=True)
      break
    else:
      for i in range(sys_num):
        cp2k_sys_dir = ''.join((cp2k_calc_dir, '/sys_', str(i)))
        task_num = process.get_task_num(cp2k_sys_dir)
        for j in range(task_num):
          cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/task_', str(j)))
          traj_num = process.get_traj_num(cp2k_sys_task_dir)
          undo_task = [index for (index,value) in enumerate(check_cp2k_run[i][j]) if value==1]
          if ( len(undo_task) != 0 and len(undo_task) < traj_num ):
            if ( job_mode == 'workstation' ):
              run_start = 0
              run_end = run_start+cp2k_job_per_node*len(host)
              if ( run_end > len(undo_task) ):
                run_end = len(undo_task)
              cycle = math.ceil(len(undo_task)/(cp2k_job_per_node*len(host)))

              for k in range(cycle):
                tot_mpi_num_list = []
                for proc_num in proc_num_per_node:
                  mpi_num_list = data_op.int_split(proc_num, cp2k_job_per_node)
                  for l in range(len(mpi_num_list)):
                    if ( mpi_num_list[l]%2 != 0 and mpi_num_list[l]>1 ):
                      mpi_num_list[l] = mpi_num_list[l]-1
                  tot_mpi_num_list.append(mpi_num_list)
                tot_mpi_num_list = data_op.list_reshape(tot_mpi_num_list)[0:(run_end-run_start+1)]
                mpi_num_str = data_op.comb_list_2_str(tot_mpi_num_list, ' ')
                task_job_list = undo_task[run_start:run_end]
                task_job_str = data_op.comb_list_2_str(task_job_list, ' ')

                cp2kfrc_parallel(task_job_str, mpi_num_str, cp2k_sys_task_dir, cp2k_job_per_node, \
                                 host_info, cp2k_env_file, cp2k_exe, parallel_exe, work_dir, ssh)

                run_start = run_start + cp2k_job_per_node*len(host)
                run_end = run_end + cp2k_job_per_node*len(host)
                if ( run_start >= len(undo_task) ):
                  run_start = len(undo_task)-1
                if ( run_end >= len(undo_task) ):
                  run_end = len(undo_task)
            elif ( job_mode == 'auto_submit' ):
              for undo_id in undo_task:
                undo_task_flag_file = ''.join((cp2k_sys_task_dir, '/traj_', str(undo_id), '/success.flag'))
                undo_task_log_file = ''.join((cp2k_sys_task_dir, '/traj_', str(undo_id), '/cp2k.out'))
                if ( os.path.exists(undo_task_flag_file) ):
                  subprocess.run('rm %s' %(undo_task_flag_file), \
                                 cwd=''.join((cp2k_sys_task_dir, '/traj_', str(undo_id))), shell=True)
                if ( os.path.exists(undo_task_log_file) ):
                  subprocess.run('rm %s' %(undo_task_log_file), \
                                 cwd=''.join((cp2k_sys_task_dir, '/traj_', str(undo_id))), shell=True)
              submit_cp2kfrc(work_dir, i, j, iter_id, undo_task, cp2k_queue[0], cp2k_exe, max_cp2k_job, \
                             cp2k_core_num, cp2k_env_file, submit_system, max_cp2k_job+i)
              while True:
                time.sleep(10)
                judge_flag = []
                judge_log = []
                for k in range(traj_num):
                  flag_file_name = ''.join((cp2k_sys_task_dir, '/traj_', str(k), '/success.flag'))
                  log_file_name = ''.join((cp2k_sys_task_dir, '/traj_', str(k), '/cp2k.out'))
                  judge_flag.append(os.path.exists(flag_file_name))
                  judge_log.append(os.path.exists(log_file_name))
                if all(judge_flag):
                  break
                else:
                  if all(judge_log):
                    time.sleep(600)
                    break

          elif ( len(undo_task) == traj_num):
            log_info.log_error('Running error: ab initio force calculations running error, please check iteration %d' %(iter_id))
            exit()

def cp2kfrc_parallel(task_job_str, mpi_num_str, cp2k_sys_task_dir, cp2k_job_per_node, \
                     host_info, cp2k_env_file, cp2k_exe, parallel_exe, work_dir, ssh):

  '''
  cp2kfrc_parallel: run cp2k force calculation in parallel.

  Args:
    task_job_str: string
      task_job_str is the string of tasks.
    mpi_num_str: string
      mpi_num_str is the string containing mpi number.
    cp2k_sys_task_dir: string
      cp2k_sys_task_dir is the directory of cp2k calculation.
    cp2k_job_per_node: int
      cp2k_job_per_node is the number of cp2k job per node.
    host_info: string
      host_info is the name of computational nodes.
    cp2k_env_file: string
      cp2k_env_file is the file name of cp2k environmnet setting.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
    parallel_exe: string
      parallel_exe is the parallel executable file.
    work_dir: string
      work_dir is working directory of DPFlow.
    ssh: bool
      ssh is whether we need to ssh.
  Returns:
    none
  '''

  import subprocess

  run_1 = '''
#! /bin/bash
task_job="%s"
mpi_num="%s"
direc=%s
parallel_exe=%s
task_job_arr=(${task_job///})
mpi_num_arr=(${mpi_num///})
num=${#task_job_arr[*]}
for ((i=0;i<=num-1;i++));
do
task_job_mpi_num_arr[i]="${task_job_arr[i]} ${mpi_num_arr[i]}"
done
''' %(task_job_str, mpi_num_str, cp2k_sys_task_dir, parallel_exe)
  if ssh:
    run_2 = '''
for i in "${task_job_mpi_num_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d -S %s --controlmaster --sshdelay 0.2 $direc/produce.sh {} $direc
''' %( cp2k_job_per_node, host_info)
  else:
    run_2 = '''
for i in "${task_job_mpi_num_arr[@]}"; do echo "$i"; done | $parallel_exe -j %d --delay 0.2 $direc/produce.sh {} $direc
''' %( cp2k_job_per_node)

  line_num = file_tools.grep_line_num('#%Module', cp2k_env_file, cp2k_sys_task_dir)
  if ( line_num == 0 ):
    set_cp2k_env = 'source %s' %(cp2k_env_file)
  else:
    set_cp2k_env = 'module load %s' %(cp2k_env_file)

  produce = '''
#! /bin/bash
%s
x=$1
direc=$2
x_arr=(${x///})
new_direc=$direc/traj_${x_arr[0]}
cd $new_direc
if [ -f "cp2k-1_0.xyz" ]; then
rm cp2k-1_0.xyz
fi
if [ -f "cp2k-1.coordLog" ]; then
rm cp2k-1.coordLog
fi
if [ -f "cp2k-1.Log" ]; then
rm cp2k-1.Log
fi
mpirun -np ${x_arr[1]} %s $new_direc/input.inp 1> $new_direc/cp2k.out 2> $new_direc/cp2k.err
converge_info=`grep "SCF run NOT converged" cp2k.out`
if [ $? -eq 0 ]; then
wfn_line=`grep -n "WFN_RESTART_FILE_NAME" input.inp`
if [ $? -eq 0 ]; then
line=`grep -n "WFN_RESTART_FILE_NAME" input.inp | awk -F ":" '{print $1}'`
sed -i ''$line's/.*/    WFN_RESTART_FILE_NAME .\/cp2k-RESTART.wfn/' input.inp
else
line=`grep -n "POTENTIAL_FILE_NAME" input.inp | awk -F ":" '{print $1}'`
sed -i ''$line' s/^/    WFN_RESTART_FILE_NAME .\/cp2k-RESTART.wfn\\n/' input.inp
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
mpirun -np ${x_arr[1]} %s $new_direc/input.inp 1> $new_direc/cp2k.out 2> $new_direc/cp2k.err
fi
cd %s
''' %(set_cp2k_env, cp2k_exe, cp2k_exe, work_dir)

  run_file_name_abs = ''.join((cp2k_sys_task_dir, '/run.sh'))
  with open(run_file_name_abs, 'w') as f:
    f.write(run_1+run_2)

  produce_file_name_abs = ''.join((cp2k_sys_task_dir, '/produce.sh'))
  with open(produce_file_name_abs, 'w') as f:
    f.write(produce)

  subprocess.run('chmod +x run.sh', cwd=cp2k_sys_task_dir, shell=True)
  subprocess.run('chmod +x produce.sh', cwd=cp2k_sys_task_dir, shell=True)
  try:
    subprocess.run("bash -c './run.sh'", cwd=cp2k_sys_task_dir, shell=True)
  except subprocess.CalledProcessError as err:
    log_info.log_error('Running error: %s command running error in %s' %(err.cmd, cp2k_sys_task_dir))

def run_cp2kfrc_ws(work_dir, iter_id, cp2k_exe, parallel_exe, cp2k_env_file, \
                   cp2k_job_per_node, proc_num_per_node, host, ssh, atoms_num_tot):

  '''
  run_cp2kfrc_ws: run cp2k force calculation for workstation mode

  Args:
    work_dir: string
      work_dir is the workding directory of DPFlow.
    iter_id: int
      iter_id is current iteration number.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
    parallel_exe: string
      parallel_exe is the parallel executable file.
    cp2k_env_file: string
      cp2k_env_file is the environment setting file of cp2k.
    cp2k_job_per_node: int
      cp2k_job_per_node is the job number of cp2k in each node.
    proc_num_per_node: 1-d int list
      proc_num_per_node is the numbers of processor in each node.
    host: 1-d string list
      host is the name of computational nodes.
    ssh: bool
      ssh is whether we need to ssh.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
  Returns:
    none
  '''

  cp2k_calc_dir = ''.join((work_dir, '/iter_', str(iter_id), '/03.cp2k_calc'))
  sys_num = process.get_sys_num(cp2k_calc_dir)

  #check generating cp2k tasks
  check_cp2k_gen(cp2k_calc_dir, sys_num)

  #run cp2k tasks
  for i in range(sys_num):
    cp2k_sys_dir = ''.join((cp2k_calc_dir, '/sys_', str(i)))
    task_num, task_dir = process.get_task_num(cp2k_sys_dir, True)
    host_name_proc = []
    for l in range(len(host)):
      host_name_proc.append(''.join((str(proc_num_per_node[l]), '/', host[l])))
    host_info = data_op.comb_list_2_str(host_name_proc, ',')

    for j in range(task_num):
      cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/', task_dir[j]))
      undo_task = find_undo_task(cp2k_sys_task_dir, atoms_num_tot[i], 'workstation')
      run_start = 0
      run_end = run_start+cp2k_job_per_node*len(host)
      if ( run_end > len(undo_task) ):
        run_end = len(undo_task)
      cycle = math.ceil(len(undo_task)/(cp2k_job_per_node*len(host)))

      for k in range(cycle):
        tot_mpi_num_list = []
        for proc_num in proc_num_per_node:
          mpi_num_list = data_op.int_split(proc_num, cp2k_job_per_node)
          for l in range(len(mpi_num_list)):
            if ( mpi_num_list[l]%2 != 0 and mpi_num_list[l]>1 ):
              mpi_num_list[l] = mpi_num_list[l]-1
          tot_mpi_num_list.append(mpi_num_list)
        tot_mpi_num_list = data_op.list_reshape(tot_mpi_num_list)[0:(run_end-run_start+1)]
        mpi_num_str = data_op.comb_list_2_str(tot_mpi_num_list, ' ')
        task_job_list = undo_task[run_start:run_end]
        task_job_str = data_op.comb_list_2_str(task_job_list, ' ')

        cp2kfrc_parallel(task_job_str, mpi_num_str, cp2k_sys_task_dir, cp2k_job_per_node, \
                         host_info, cp2k_env_file, cp2k_exe, parallel_exe, work_dir, ssh)

        run_start = run_start + cp2k_job_per_node*len(host)
        run_end = run_end + cp2k_job_per_node*len(host)
        if ( run_start >= len(undo_task) ):
          run_start = len(undo_task)-1
        if ( run_end >= len(undo_task) ):
          run_end = len(undo_task)

  #check running cp2k tasks
  run_undo_cp2kfrc(work_dir, iter_id, cp2k_env_file, cp2k_exe, atoms_num_tot, 'workstation', \
                   cp2k_job_per_node, host, ssh, proc_num_per_node, parallel_exe, None, None, None, None)
  
def submit_cp2kfrc(work_dir, sys_id, task_id, iter_id, undo_task, cp2k_queue, cp2k_exe, \
                   max_cp2k_job, cp2k_core_num, cp2k_env_file, submit_system, cycle_id, return_job_id=False):

  '''
  submit_cp2kfrc: submit discrete cp2k force jobs to remote host.

  Args:
    cp2k_sys_task_dir: string
      cp2k_sys_task_dir is the directory of cp2k for one system and one task
    sys_id: int
      sys_id is the id of system.
    task_id: int
      task_id is the id of task.
    iter_id: int
      iter_id is the iteration id.
    undo_task: 1-d int list
      undo_task are id of uncompleted lammps force tasks.
    cp2k_queue: string
      cp2k_queue is the queue name of cp2k job.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
    max_cp2k_job: int
      max_cp2k_job is the maximum number of sumbition for cp2k jobs.
    cp2k_core_num: int
      cp2k_core_num is the number of cores for each cp2k job.
    cp2k_env_file: string
      cp2k_env_file is the file name of cp2k environmnet setting.
    submit_system: string
      submit_system is the submition system.
    cycle_id: int
      cycle_id is the id of cp2k submition.
  Returns:
    none
  '''

  import numpy as np

  cp2k_sys_task_dir = ''.join((work_dir, '/iter_', str(iter_id), '/03.cp2k_calc/sys_', \
                               str(sys_id), '/task_', str(task_id)))
  rand_int = np.random.randint(10000000000)
  job_label = ''.join(('cp2k_', str(rand_int)))

  line_num = file_tools.grep_line_num('#%Module', cp2k_env_file, cp2k_sys_task_dir)
  if ( line_num == 0 ):
    set_cp2k_env = 'source %s' %(cp2k_env_file)
  else:
    set_cp2k_env = 'module load %s' %(cp2k_env_file)

  task_index = data_op.comb_list_2_str(undo_task, ' ')
  
  submit_file_name_abs = ''.join((cp2k_sys_task_dir, '/cp2k_', str(cycle_id), '.sub'))
  if ( submit_system == 'lsf' ):

    script_1 = gen_shell_str.gen_lsf_normal(cp2k_queue, cp2k_core_num, job_label)
    script_2 = gen_shell_str.gen_cd_lsfcwd()
    script_3 = gen_shell_str.gen_cp2k_script(set_cp2k_env, cp2k_sys_task_dir, task_index, cp2k_core_num, cp2k_exe)
    with open(submit_file_name_abs, 'w') as f:
      f.write(script_1+script_2+script_3)
    subprocess.run('bsub < ./cp2k_%d.sub'%(cycle_id), cwd=cp2k_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'pbs' ):

    script_1 = gen_shell_str.gen_pbs_normal(cp2k_queue, cp2k_core_num, 0, job_label)
    script_2 = gen_shell_str.gen_cd_pbscwd()
    script_3 = gen_shell_str.gen_cp2k_script(set_cp2k_env, cp2k_sys_task_dir, task_index, cp2k_core_num, cp2k_exe)
    with open(submit_file_name_abs, 'w') as f:
      f.write(script_1+script_2+script_3)
    subprocess.run('qsub ./cp2k_%d.sub'%(cycle_id), cwd=cp2k_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

  if ( submit_system == 'slurm' ):

    script_1 = gen_shell_str.gen_slurm_normal(cp2k_queue, cp2k_core_num, job_label)
    script_2 = gen_shell_str.gen_cp2k_script(set_cp2k_env, cp2k_sys_task_dir, task_index, cp2k_core_num, cp2k_exe)
    with open(submit_file_name_abs, 'w') as f:
      f.write(script_1+script_2)
    subprocess.run('sbatch ./cp2k_%d.sub'%(cycle_id), cwd=cp2k_sys_task_dir, shell=True, stdout=subprocess.DEVNULL)

  if return_job_id:
    job_id = process.get_job_id(work_dir, submit_system, 'cp2k_', rand_int)
    return job_id

def run_cp2kfrc_as(work_dir, iter_id, cp2k_queue, cp2k_exe, max_cp2k_job, \
                   cp2k_core_num, cp2k_env_file, submit_system, atoms_num_tot):

  '''
  run_cp2kfrc_ws: run cp2k force calculation for auto_submit mode

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    iter_id: int
      iter_id is the iteration id.
    cp2k_queue: string
      cp2k_queue is the queue name of cp2k job.
    cp2k_exe: string
      cp2k_exe is the cp2k executable file.
    max_cp2k_job: int
      max_cp2k_job is the maximum number of sumbition for cp2k jobs.
    cp2k_core_num: int
      cp2k_core_num is the number of cores for each cp2k job.
    cp2k_env_file: string
      cp2k_env_file is the file name of cp2k environmnet setting.
    submit_system: string
      submit_system is the submition system.
    atoms_num_tot: 1-d dictionary, dim = num of lammps systems
      example: {1:192,2:90}
  Returns:
    None
  '''

  cp2k_calc_dir = ''.join((work_dir, '/iter_', str(iter_id), '/03.cp2k_calc'))
  sys_num = process.get_sys_num(cp2k_calc_dir)

  #check generating cp2k tasks
  check_cp2k_gen(cp2k_calc_dir, sys_num)

  cp2k_queue = cp2k_queue*max_cp2k_job
  cp2k_queue_sub = cp2k_queue[0:max_cp2k_job]

  #run cp2k tasks
  for i in range(sys_num):
    cp2k_sys_dir = ''.join((cp2k_calc_dir, '/sys_', str(i)))
    task_num, task_dir = process.get_task_num(cp2k_sys_dir, True)

    for j in range(task_num):
      cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/', task_dir[j]))
      traj_num = process.get_traj_num(cp2k_sys_task_dir)
      undo_task = find_undo_task(cp2k_sys_task_dir, atoms_num_tot[i], 'auto_submit')
      if ( len(undo_task) != 0 ):
        for undo_id in undo_task:
          undo_task_flag_file = ''.join((cp2k_sys_task_dir, '/traj_', str(undo_id), '/success.flag'))
          if ( os.path.exists(undo_task_flag_file) ):
            subprocess.run('rm %s' %(undo_task_flag_file), \
                           cwd=''.join((cp2k_sys_task_dir, '/traj_', str(undo_id))), shell=True)

        cp2k_job_per_submit = math.ceil(len(undo_task)/max_cp2k_job)
        undo_task_split = data_op.list_split(undo_task, cp2k_job_per_submit)
        undo_task_parts = []
        for k in undo_task_split:
          undo_task_parts.append(k)
        job_id = []
        failure_id = []
        for k in range(len(undo_task_parts)):
          job_id_part = submit_cp2kfrc(work_dir, i, j, iter_id, undo_task_parts[k], cp2k_queue[k], \
                        cp2k_exe, max_cp2k_job, cp2k_core_num, cp2k_env_file, submit_system, k, True)
          if ( job_id_part > 0 ):
            job_id.append(job_id_part)
          else:
            failure_id.append(k)
        if ( len(job_id) == len(undo_task_parts) ):
          str_print = 'Success: submit cp2k job for system %d task %d in iteration %d with job id %s' %(i, j, iter_id, data_op.comb_list_2_str(job_id, ' '))
          str_print = data_op.str_wrap(str_print, 80, '  ')
          print (str_print, flush=True)

          while True:
            time.sleep(10)
            judge_flag = []
            judge_log = []
            for k in range(traj_num):
              flag_file_name = ''.join((cp2k_sys_task_dir, '/traj_', str(k), '/success.flag'))
              log_file_name = ''.join((cp2k_sys_task_dir, '/traj_', str(k), '/cp2k.out'))
              judge_flag.append(os.path.exists(flag_file_name))
              judge_log.append(os.path.exists(log_file_name))
            if all(judge_flag):
              break
            else:
              if all(judge_log):
                time.sleep(600)
                break
        else:
          log_info.log_error('Fail to submit lammps force job for system %d task %d in iteration %d' \
                           %(i, j, iter_id))
          exit()

  run_undo_cp2kfrc(work_dir, iter_id, cp2k_env_file, cp2k_exe, atoms_num_tot, 'auto_submit', None, \
                   None, None, None, None, cp2k_queue, max_cp2k_job, cp2k_core_num, submit_system)

if __name__ == '__main__':
  from collections import OrderedDict
  from DPFlow.tools import read_input
  from DPFlow.deepff import cp2k_run
  from DPFlow.deepff import check_deepff

  work_dir = '/home/lujunbo/WORK/Deepmd/DPFlow/co2/md_mtd'
  deepff_key = ['deepmd', 'lammps', 'cp2k', 'model_devi', 'environ']
  deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic = \
  read_input.dump_info(work_dir, 'input.inp', deepff_key)
  proc_num = 4
  deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic = \
  check_deepff.check_inp(deepmd_dic, lammps_dic, cp2k_dic, model_devi_dic, environ_dic, proc_num)

  #Test gen_cp2kfrc_file function
  coord = [[['O','-9.64','-0.71','5.80'],['H','-10.39','-1.31','6.15'],['H','-8.89','-35.4','6.37']],
            [['O','-2.64','-7.14','5.52'],['H','-2.89','-6.23','5.10'],['H','-1.70','-7.36','5.28']]]
  box = [[['10.0','0.0','0.0'],['0.0','10.0','0.0'],['0.0','0.0','10.0']],
         [['10.0','0.0','0.0'],['0.0','10.0','0.0'],['0.0','0.0','10.0']]]
#  cp2k_run.gen_cp2kfrc_file(cp2k_dic, work_dir, 1, 0, coord, box)

  #Test gen_cp2k_task function
  atoms_type_multi_sys = {0: {'C': 1, 'O': 2}, 1: {'C': 1, 'O': 2}}
  atoms_num_tot = {0:3, 1:3}
  struct_index = OrderedDict([(0, OrderedDict([(0, [237, 264, 275, 291, 331, 367, 422])])), (1, OrderedDict([(0, [])]))])
  conv_new_data_num = 5
  choose_new_data_num_limit = 100
  cp2k_run.gen_cp2k_task(cp2k_dic, work_dir, 17, atoms_type_multi_sys, atoms_num_tot, \
                         struct_index, conv_new_data_num, choose_new_data_num_limit, False)

  #Test run_cp2kfrc function
#  cp2k_run.run_cp2kfrc(work_dir, 0, environ_dic)
