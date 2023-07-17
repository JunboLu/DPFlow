#! /usr/env/bin python

import os
import json
import copy
import numpy as np
import operator
from collections import OrderedDict
from DPFlow.tools import call
from DPFlow.tools import data_op
from DPFlow.tools import log_info
from DPFlow.deepff import check_deepff
from DPFlow.deepff import load_data
from DPFlow.deepff import gen_deepmd_task
from DPFlow.deepff import deepmd_run
from DPFlow.deepff import gen_lammps_task
from DPFlow.deepff import lammps_md_run
from DPFlow.deepff import lammps_frc_run
from DPFlow.deepff import model_devi
from DPFlow.deepff import dp_test
from DPFlow.deepff import gen_cp2k_task
from DPFlow.deepff import cp2k_run
from DPFlow.deepff import sys_info
from DPFlow.deepff import write_data
from DPFlow.deepff import process

def model_devi_iter(work_dir, inp_file, deepmd_dic, lammps_dic, cp2k_dic, active_learn_dic, \
                    environ_dic, init_train_data, init_data_num, tot_atoms_type_dic):

  '''
  model_devi_iter: run active learning iterations with model deviation.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    inp_file: string
      inp_file is the input file of DPFlow.
    deepmd_dic: dictionary
      deepmd_dic contains keywords used in deepmd.
    lammps_dic: dictionary
      lammpd_dic contains keywords used in lammps.
    cp2k_dic: dictionary
      cp2k_dic contains keywords used in cp2k.
    active_learn_dic: dictionary
      active_learn_dic contains keywords used in active learning.
    environ_dic: dictionary
      environ_dic contains keywords used in environment.
    init_train_data: 1-d string list
      init_train_data contains initial training data directories.
    init_data_num: int
      init_data_num is the data number for initial training.
    tot_atoms_type_dic: dictionary
      tot_atoms_type_dic is the atoms type dictionary.
  Returns:
    none
  '''

  proc_num, proc_num_per_node, host, ssh = sys_info.get_host(work_dir)
  parallel_exe = environ_dic['parallel_exe']
  device = sys_info.analyze_gpu(host, ssh, parallel_exe, work_dir) 

  if ( len(host) > 1 and len(device[0]) >= 1 ):
    judge_device = []
    for i in range(len(host)):
      for j in range(len(host)):
        if ( j>i ):
          judge_device.operator.eq(device[i], device[j])
    if not all(judge_device):
      log_info.log_error('Resource error: please submit your jobs to nodes with same number of availale GPU device')
      exit()

  analyze_gpu = environ_dic['analyze_gpu']
  if ( ssh and len(device[0]) >= 1 ) :
    host_ssh, device_ssh = sys_info.get_remote_gpu()
    if analyze_gpu:
      device_ssh = sys_info.analyze_gpu([host_ssh], ssh, parallel_exe, work_dir)[0]

  max_iter = active_learn_dic['max_iter']
  restart_iter = active_learn_dic['restart_iter']

  if ( restart_iter == 0 ):
    data_num = init_data_num
  else:
    data_num = active_learn_dic['data_num']
  restart_stage = active_learn_dic['restart_stage']

  cmd = "ls | grep %s" %("'iter_'")
  iter_info = call.call_returns_shell(work_dir, cmd)
  if ( restart_iter == 0 and restart_stage == 0 and len(iter_info) > 0 ):
    iter_0_dir = ''.join((work_dir, '/iter_0'))
    dir_num = len(call.call_returns_shell(iter_0_dir, "ls -ll |awk '/^d/ {print $NF}'"))
    if ( dir_num > 1 ):
      log_info.log_error('There are iteration directories in %s. If you restart the deepmd train in iteration 0, \
                          you could ignore this warning; if not, please use DPFlow.restart file in %s as input.' \
                          %(work_dir, work_dir), 'Warning')

  atom_mass_dic = deepmd_dic['model']['atom_mass']
  numb_test = deepmd_dic['training']['numb_test']
  model_type = deepmd_dic['training']['model_type']
  neuron = deepmd_dic['training']['neuron']
  shuffle_data = deepmd_dic['training']['shuffle_data']
  train_stress = deepmd_dic['training']['train_stress']
  use_prev_model = deepmd_dic['training']['use_prev_model']

  nsteps = int(lammps_dic['nsteps'])
  judge_freq = int(active_learn_dic['judge_freq'])
  conv_new_data_num = int(nsteps/judge_freq*0.04)
  choose_new_data_num_limit = active_learn_dic['choose_new_data_num_limit']
  success_force_conv = active_learn_dic['success_force_conv']
  max_force_conv = active_learn_dic['max_force_conv']
  active_learn_steps = int(nsteps/judge_freq)+1

  cp2k_exe = environ_dic['cp2k_exe']
  cp2k_env_file = environ_dic['cp2k_env_file']
  parallel_exe = environ_dic['parallel_exe']
  cuda_dir = environ_dic['cuda_dir']
  dp_version = environ_dic['dp_version']
  cp2k_job_per_node = environ_dic['cp2k_job_per_node']
  lmp_md_job_per_node = environ_dic['lmp_md_job_per_node']
  lmp_frc_job_per_node = environ_dic['lmp_frc_job_per_node']
  job_mode = environ_dic['job_mode']
  submit_system = environ_dic['submit_system']
  dp_queue = environ_dic['dp_queue']
  lmp_queue = environ_dic['lmp_queue']
  cp2k_queue = environ_dic['cp2k_queue']
  max_dp_job = environ_dic['max_dp_job']
  max_lmp_job = environ_dic['max_lmp_job']
  max_cp2k_job = environ_dic['max_cp2k_job']
  dp_core_num = environ_dic['dp_core_num']
  lmp_core_num = environ_dic['lmp_core_num']
  lmp_gpu_num = environ_dic['lmp_gpu_num']
  dp_gpu_num = environ_dic['dp_gpu_num']
  cp2k_core_num = environ_dic['cp2k_core_num']

  lmp_mpi_num_per_job = int(proc_num_per_node[0]/2)
  lmp_omp_num_per_job = 2

  dp_path = sys_info.get_dp_path(work_dir)
  lmp_exe, lmp_path = sys_info.get_lmp_path(work_dir)
  mpi_path = sys_info.get_mpi_path(work_dir)

  #np.random.seed(1234567890)

  for i in range(restart_iter, max_iter, 1):

    print (''.join(('iter_', str(i))).center(80,'*'), flush=True)

    #Generate iteration directory
    iter_restart = ''.join(('iter_', str(i)))
    iter_restart_dir = ''.join((work_dir, '/', iter_restart))

    if ( not os.path.exists(iter_restart_dir) ):
      cmd = "mkdir %s" % (iter_restart)
      call.call_simple_shell(work_dir, cmd)

    if ( restart_stage == 0 ):
      #Perform deepmd calculation
      write_data.write_restart_inp(inp_file, i, 0, data_num, work_dir)
      print ('Step 1: deepmd-kit tasks', flush=True)

      #For different model_type, seed and neuron are different.
      if ( model_type == 'use_seed' ):
        if ( 'seed_num' in deepmd_dic['training'].keys() ):
          seed_num = int(deepmd_dic['training']['seed_num'])
        else:
          seed_num = 4
        descr_seed = []
        fit_seed = []
        tra_seed = []
        for j in range(seed_num):
          descr_seed.append(np.random.randint(10000000000))
          fit_seed.append(np.random.randint(10000000000))
          tra_seed.append(np.random.randint(10000000000))

      if ( model_type == 'use_node' ):
        descr_seed = []
        fit_seed = []
        tra_seed = []

        for j in range(len(neuron)):
          descr_seed.append(np.random.randint(10000000000))
          fit_seed.append(np.random.randint(10000000000))
          tra_seed.append(np.random.randint(10000000000))

      gen_deepmd_task.gen_deepmd_model_task(deepmd_dic, work_dir, i, init_train_data, numb_test, descr_seed, fit_seed, \
                                            tra_seed, neuron, model_type, data_num, tot_atoms_type_dic, dp_version)

      if ( job_mode == 'workstation' ):
        if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
          deepmd_run.run_deepmd_ws(work_dir, i, use_prev_model, parallel_exe, dp_path, [host_ssh], \
                                   ssh, [device_ssh], cuda_dir, dp_version, analyze_gpu, 0)
        else:
          deepmd_run.run_deepmd_ws(work_dir, i, use_prev_model, parallel_exe, dp_path, \
                                   host, ssh, device, cuda_dir, dp_version, analyze_gpu, 0)
      elif ( job_mode == 'auto_submit' ):
        deepmd_run.run_deepmd_as(work_dir, i, dp_queue, dp_core_num, dp_gpu_num, max_dp_job, submit_system, \
                                 use_prev_model, dp_path, cuda_dir, dp_version, analyze_gpu, 0)

      write_data.write_restart_inp(inp_file, i, 1, data_num, work_dir)

      failure_model = process.check_deepff_run(work_dir, i, dp_version)
      if ( len(failure_model) == 0 ):
        pass
      else:
        print ('Warning'.center(80,'*'), flush=True)
        for model_id in failure_model:
          str_print = 'The training is failed as force is fluctuated in No.%d model in No.%d iteration' %(model_id, i)
          str_print = data_op.str_wrap(str_print, 80, '')
          print (str_print, flush=True)
        exit()

    if ( restart_stage == 0 or restart_stage == 1 ):
      #Perform lammps calculations
      print ('Step 2: lammps tasks', flush=True)

      gen_lammps_task.gen_lmpmd_task(lammps_dic, work_dir, i, atom_mass_dic, tot_atoms_type_dic, dp_version)

      if ( job_mode == 'workstation' ):
        if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
          lammps_md_run.run_lmpmd_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, mpi_path, \
                                     lmp_md_job_per_node, lmp_mpi_num_per_job, lmp_omp_num_per_job, \
                                     proc_num_per_node, [host_ssh], [device_ssh], analyze_gpu)
        else:
          lammps_md_run.run_lmpmd_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, mpi_path, \
                                     lmp_md_job_per_node, lmp_mpi_num_per_job, lmp_omp_num_per_job, \
                                     proc_num_per_node, host, device, analyze_gpu)
      elif ( job_mode == 'auto_submit' ):
        lammps_md_run.run_lmpmd_as(work_dir, i, lmp_queue, max_lmp_job, lmp_core_num, lmp_gpu_num, \
                                   submit_system, lmp_path, lmp_exe, mpi_path, analyze_gpu)

      write_data.write_restart_inp(inp_file, i, 2, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 ):
      #Perform lammps force calculations
      if ( restart_stage == 2 ):
        print ('Step 2: lammps tasks', flush=True)
      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)
      gen_lammps_task.gen_lmpfrc_file(work_dir, i, atom_mass_dic, atoms_num_tot, \
                                      atoms_type_multi_sys, use_bias_tot, 'model_devi', dp_version)
      if ( job_mode == 'workstation' ):
        if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
          lammps_frc_run.run_lmpfrc_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, \
                                       lmp_frc_job_per_node, [host_ssh], [device_ssh], atoms_num_tot, \
                                       use_bias_tot, 'model_devi')
        else:
          lammps_frc_run.run_lmpfrc_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, \
                                       lmp_frc_job_per_node, host, device, atoms_num_tot, \
                                       use_bias_tot, 'model_devi')
      elif ( job_mode == 'auto_submit' ):
        #It is better to use cp2k queue for lammps force calculation.
        #We find it will be very slow when we use gpu queue for lammps force calculation.
        lammps_frc_run.run_lmpfrc_as(work_dir, i, 'model_devi', use_bias_tot, lmp_path, lmp_exe, \
                                     cp2k_queue, cp2k_core_num, max_cp2k_job, parallel_exe, \
                                     submit_system, atoms_num_tot)

      write_data.write_restart_inp(inp_file, i, 3, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 or restart_stage == 3 ):
      #Get force-force correlation and then choose new structures
      print ('step 3: model deviation', flush=True)
      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = \
      process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)

      struct_index, success_ratio_sys, success_ratio, success_devi_ratio = \
      model_devi.choose_lmp_str(work_dir, i, atoms_type_multi_sys, use_bias_tot, \
                                success_force_conv, max_force_conv)

      for j in range(len(success_ratio_sys)):
        print ('  The accurate ratio for system %d in iteration %d is %.2f%%' \
               %(j, i, success_ratio_sys[j]*100), flush=True)

      print ('  The accurate ratio for whole %d systems in iteration %d is %.2f%%' \
             %(sys_num, i, success_ratio*100), flush=True)
      print ('  The accurate deviation ratio for whole %d systems in iteration %d is %.2f%%' \
             %(sys_num, i, success_devi_ratio*100), flush=True)

      if ( min(success_ratio_sys) >= 0.95 and success_ratio+success_devi_ratio > 0.99 ):
        print (''.center(80,'*'), flush=True)
        print ('Cheers! deepff is converged!', flush=True)
        if ( i != 0 ):
          write_data.write_active_data(work_dir, i, tot_atoms_type_dic)
        exit()

      total_task_num = 0
      for sys_id in struct_index:
        total_task_num_sys = 0
        for task_id in struct_index[sys_id]:
          total_task_num_sys = total_task_num_sys+len(struct_index[sys_id][task_id])
        total_task_num = total_task_num+total_task_num_sys
      if ( total_task_num == 0 ):
        log_info.log_error('Warning: No selected structure for cp2k calculations, check the deepmd training.')
        exit()

      print ('Step 4: cp2k tasks', flush=True)
      gen_cp2k_task.gen_cp2k_task(cp2k_dic, work_dir, i, atoms_type_multi_sys, atoms_num_tot, \
                                  struct_index, conv_new_data_num, choose_new_data_num_limit, \
                                  train_stress, 'model_devi', success_ratio, success_devi_ratio)

      write_data.write_restart_inp(inp_file, i, 4, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 \
         or restart_stage == 3 or restart_stage == 4 ):
      if ( restart_stage == 4 ):
        print ('Step 4: cp2k tasks', flush=True)
      #Perform cp2k calculation
      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = \
      process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)

      if ( job_mode == 'workstation' ):
        cp2k_run.run_cp2kfrc_ws(work_dir, i, cp2k_exe, parallel_exe, cp2k_env_file, \
                                cp2k_job_per_node, proc_num_per_node, host, ssh, atoms_num_tot)
      elif ( job_mode == 'auto_submit' ):
        cp2k_run.run_cp2kfrc_as(work_dir, i, cp2k_queue, cp2k_exe, max_cp2k_job, \
                                cp2k_core_num, cp2k_env_file, submit_system, atoms_num_tot)

      #Dump new data of cp2k
      for j in range(sys_num):
        cp2k_sys_dir = ''.join((work_dir, '/iter_', str(i), '/03.cp2k_calc/sys_', str(j)))
        task_num, task_dir = process.get_task_num(cp2k_sys_dir, True)
        for k in range(task_num):
          cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/', task_dir[k]))
          traj_num = process.get_traj_num(cp2k_sys_task_dir)
          if ( traj_num != 0 ):
            data_dir = ''.join((cp2k_sys_task_dir, '/data'))
            if ( not os.path.exists(data_dir) ):
              cmd = "mkdir %s" % ('data')
              call.call_simple_shell(cp2k_sys_task_dir, cmd)
            total_index = data_op.gen_list(0, traj_num-1, 1)
            total_index_array = np.array(total_index)
            np.random.shuffle(total_index_array)
            choosed_index = list(total_index_array[0:traj_num])
            load_data.load_data_from_sepfile(cp2k_sys_task_dir, data_dir, 'traj_', \
                                             'cp2k', tot_atoms_type_dic, choosed_index)
            energy_array, coord_array, frc_array, box_array, virial_array = load_data.read_raw_data(data_dir)
            train_data_num, test_data_num = load_data.raw_data_to_set(1, shuffle_data, data_dir, energy_array, \
                                                                    coord_array, frc_array, box_array, virial_array)
            if ( test_data_num > numb_test ):
              data_num.append(train_data_num)
            if ( test_data_num < numb_test and success_ratio < float((active_learn_steps-train_data_num)/active_learn_steps) ):
              log_info.log_error('Warning: little selected structures, check the deepmd training.')
              exit()

      print ('  Success: dump new raw data of cp2k', flush=True)
      restart_stage = 0

    write_data.write_restart_inp(inp_file, i+1, 0, data_num, work_dir)

    if ( i == max_iter-1 ):
      log_info.log_error('Active learning does not converge')
      write_data.write_active_data(work_dir, i+1, tot_atoms_type_dic)

def dp_test_iter(work_dir, inp_file, deepmd_dic, lammps_dic, active_learn_dic, cp2k_dic, environ_dic):

  '''
  dp_test_iter: run active learning iterations with deepmd test.

  Args:
    work_dir: string
      work_dir is working directory of DPFlow.
    inp_file: string
      inp_file is the input file of DPFlow.
    deepmd_dic: dictionary
      deepmd_dic contains keywords used in deepmd.
    lammps_dic: dictionary
      lammpd_dic contains keywords used in lammps.
    active_learn_dic: dictionary
      active_learn_dic contains keywords used in active learning.
    cp2k_dic: dictionary
      cp2k_dic contains keywords used in cp2k.
    environ_dic: dictionary
      environ_dic contains keywords used in environment.
  Returns:
    none
  '''

  proc_num, proc_num_per_node, host, ssh = sys_info.get_host(work_dir)
  parallel_exe = environ_dic['parallel_exe']
  device = sys_info.analyze_gpu(host, ssh, parallel_exe, work_dir)

  if ( len(host) > 1 and len(device[0]) >= 1 ):
    judge_device = []
    for i in range(len(host)):
      for j in range(len(host)):
        if ( j>i ):
          judge_device.operator.eq(device[i], device[j])
    if not all(judge_device):
      log_info.log_error('Resource error: please submit your jobs to nodes with same number of availale GPU device')
      exit()

  analyze_gpu = environ_dic['analyze_gpu']
  if ( ssh and len(device[0]) >= 1 ) :
    host_ssh, device_ssh = sys_info.get_remote_gpu()
    if analyze_gpu:
      device_ssh = sys_info.analyze_gpu([host_ssh], ssh, parallel_exe, work_dir)[0]

  max_iter = active_learn_dic['max_iter']
  restart_iter = active_learn_dic['restart_iter']
  data_num = active_learn_dic['data_num']
  restart_stage = active_learn_dic['restart_stage']

  cmd = "ls | grep %s" %("'iter_'")
  iter_info = call.call_returns_shell(work_dir, cmd)
  if ( restart_iter == 0 and restart_stage == 0 and len(iter_info) > 0 ):
    iter_0_dir = ''.join((work_dir, '/iter_0'))
    dir_num = len(call.call_returns_shell(iter_0_dir, "ls -ll |awk '/^d/ {print $NF}'"))
    if ( dir_num > 1 ):
      log_info.log_error('There are iteration directories in %s, please use DPFlow.restart file in %s as input.' \
                          %(work_dir, work_dir), 'Warning')
      exit()

  init_dpff_dir = deepmd_dic['init_dpff_dir']
  use_prev_model = deepmd_dic['use_prev_model']
  train_stress = deepmd_dic['train_stress']
  deepmd_inp_file = ''.join((init_dpff_dir, '/input.json'))
  if ( not os.path.exists(deepmd_inp_file) ):
    log_info.log_error("Running error: %s does not exist" %(deepmd_inp_file))
    exit()

  with open(deepmd_inp_file, 'r') as f:
    deepmd_dic_json = json.load(f)

  atom_mass_dic = deepmd_dic['atom_mass']
  shuffle_data = deepmd_dic['shuffle_data']
  numb_test = deepmd_dic_json['training']['numb_test']
  tot_atoms_type = deepmd_dic_json['model']['type_map']
  tot_atoms_type_dic = OrderedDict()
  for i in range(len(tot_atoms_type)):
    tot_atoms_type_dic[tot_atoms_type[i]] = i

  nsteps = int(lammps_dic['nsteps'])
  change_init_str = lammps_dic['change_init_str']
  judge_freq = int(active_learn_dic['judge_freq'])
  conv_new_data_num = int(nsteps/judge_freq*0.04)
  choose_new_data_num_limit = active_learn_dic['choose_new_data_num_limit']
  data_num = active_learn_dic['data_num']
  success_force_conv = active_learn_dic['success_force_conv']
  max_force_conv = active_learn_dic['max_force_conv']
  energy_conv = active_learn_dic['energy_conv']

  cp2k_exe = environ_dic['cp2k_exe']
  cp2k_env_file = environ_dic['cp2k_env_file']
  parallel_exe = environ_dic['parallel_exe']
  cuda_dir = environ_dic['cuda_dir']
  dp_version = environ_dic['dp_version']
  cp2k_job_per_node = environ_dic['cp2k_job_per_node']
  lmp_md_job_per_node = environ_dic['lmp_md_job_per_node']
  lmp_frc_job_per_node = environ_dic['lmp_frc_job_per_node']
  lmp_mpi_num_per_job = int(proc_num_per_node[0]/2)
  lmp_omp_num_per_job = 2
  job_mode = environ_dic['job_mode']
  submit_system = environ_dic['submit_system']
  dp_queue = environ_dic['dp_queue']
  lmp_queue = environ_dic['lmp_queue']
  cp2k_queue = environ_dic['cp2k_queue']
  max_dp_job = environ_dic['max_dp_job']
  max_lmp_job = environ_dic['max_lmp_job']
  max_cp2k_job = environ_dic['max_cp2k_job']
  dp_core_num = environ_dic['dp_core_num']
  lmp_core_num = environ_dic['lmp_core_num']
  lmp_gpu_num = environ_dic['lmp_gpu_num']
  dp_gpu_num = environ_dic['dp_gpu_num']
  cp2k_core_num = environ_dic['cp2k_core_num']

  dp_path = sys_info.get_dp_path(work_dir)
  lmp_exe, lmp_path = sys_info.get_lmp_path(work_dir)
  mpi_path = sys_info.get_mpi_path(work_dir)

  for i in range(restart_iter, max_iter, 1):

    print (''.join(('iter_', str(i))).center(80,'*'), flush=True)

    #Generate iteration directory
    iter_restart = ''.join(('iter_', str(i)))
    iter_restart_dir = ''.join((work_dir, '/', iter_restart))

    if ( not os.path.exists(iter_restart_dir) ):
      cmd = "mkdir %s" % (iter_restart)
      call.call_simple_shell(work_dir, cmd)

    if ( restart_stage == 0 ):
      #Perform deepmd calculation
      write_data.write_restart_inp(inp_file, i, 0, data_num, work_dir)
      print ('Step 1: deepmd-kit tasks', flush=True)

      gen_deepmd_task.gen_deepmd_test_task(deepmd_dic, work_dir, i, data_num, tot_atoms_type_dic)
      if ( i>0 ):
        if ( job_mode == 'workstation' ):
          if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
            deepmd_run.run_deepmd_ws(work_dir, i, use_prev_model, parallel_exe, dp_path, [host_ssh], ssh, \
                                     [device_ssh], cuda_dir, dp_version, analyze_gpu, 1)
          else:
            deepmd_run.run_deepmd_ws(work_dir, i, use_prev_model, parallel_exe, dp_path, host, ssh, \
                                     device, cuda_dir, dp_version, analyze_gpu, 1)
        elif ( job_mode == 'auto_submit' ):
          deepmd_run.run_deepmd_as(work_dir, i, dp_queue, dp_core_num, dp_gpu_num, max_dp_job, submit_system, \
                                   use_prev_model, dp_path, cuda_dir, dp_version, analyze_gpu, 1)

      else:
        str_print = 'Success: the initial deep potential file is copied in %s' \
                    %(''.join((work_dir, '/iter_0/01.train/0')))
        str_print = data_op.str_wrap(str_print, 80, '  ')
        print (str_print, flush=True)
      write_data.write_restart_inp(inp_file, i, 1, data_num, work_dir)
      if ( i>0 ):
        failure_model = process.check_deepff_run(work_dir, i, dp_version)
        if ( len(failure_model) == 0 ):
          pass
        else:
          print ('Warning'.center(80,'*'), flush=True)
          for model_id in failure_model:
            str_print = 'The training is failed as force is fluctuated in No.%d model in No.%d iteration' %(model_id, i)
            str_print = data_op.str_wrap(str_print, 80, '  ')
            print (str_print, flush=True)
          exit()

    if ( restart_stage == 0 or restart_stage == 1 ):
      #Perform lammps calculations
      print ('Step 2: lammps tasks', flush=True)
      gen_lammps_task.gen_lmpmd_task(lammps_dic, work_dir, i, atom_mass_dic, tot_atoms_type_dic, dp_version)

      if ( job_mode == 'workstation' ):
        if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
          lammps_md_run.run_lmpmd_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, mpi_path, \
                                     lmp_md_job_per_node, lmp_mpi_num_per_job, lmp_omp_num_per_job, \
                                     proc_num_per_node, [host_ssh], [device_ssh], analyze_gpu)
        else:
          lammps_md_run.run_lmpmd_ws(work_dir, i, lmp_path, lmp_exe, parallel_exe, mpi_path, \
                                     lmp_md_job_per_node, lmp_mpi_num_per_job, lmp_omp_num_per_job, \
                                     proc_num_per_node, host, device, analyze_gpu)
      elif ( job_mode == 'auto_submit' ):
        lammps_md_run.run_lmpmd_as(work_dir, i, lmp_queue, max_lmp_job, lmp_core_num, lmp_gpu_num, \
                                   submit_system, lmp_path, lmp_exe, mpi_path, analyze_gpu)

    write_data.write_restart_inp(inp_file, i, 2, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 ):
      #Perform lammps force calculations
      if ( restart_stage == 2 ):
        print ('Step 2: lammps tasks', flush=True)

      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = \
      process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)
      gen_lammps_task.gen_lmpfrc_file(work_dir, i, atom_mass_dic, atoms_num_tot, \
                                      atoms_type_multi_sys, use_bias_tot, 'dp_test', dp_version)
      if ( job_mode == 'workstation' ):
        if ( ssh and len(host) >= 1 and len(device[0]) > 0 ):
          lammps_frc_run.run_lmpfrc_ws(work_dir, i, lmp_path, lmp_exe, mpi_path, parallel_exe, \
                                       lmp_frc_job_per_node, [host_ssh], [device_ssh], atoms_num_tot, \
                                       use_bias_tot, 'dp_test')
        else:
          lammps_frc_run.run_lmpfrc_ws(work_dir, i, lmp_path, lmp_exe, mpi_path, parallel_exe, \
                                       lmp_frc_job_per_node, host, device, atoms_num_tot, use_bias_tot, 'dp_test')
      elif ( job_mode == 'auto_submit' ):
        lammps_frc_run.run_lmpfrc_as(work_dir, i, 'dp_test', use_bias_tot, lmp_path, lmp_exe, \
                                     cp2k_queue, cp2k_core_num, max_cp2k_job, parallel_exe, \
                                     submit_system, atoms_num_tot)

      write_data.write_restart_inp(inp_file, i, 3, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 or restart_stage == 3 ):
      print ('Step 3: cp2k tasks', flush=True)
      #Perform cp2k calculation
      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = \
      process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)
      total_index = OrderedDict()
      for j in range(sys_num):
        total_index_j = OrderedDict()
        lmp_sys_dir = ''.join((work_dir, '/iter_', str(i), '/02.lammps_calc/sys_', str(j)))
        task_num = process.get_task_num(lmp_sys_dir)
        for k in range(task_num):
          lmp_sys_task_data_dir = ''.join((lmp_sys_dir, '/task_', str(k), '/data'))
          data_file_num = process.get_data_num(lmp_sys_task_data_dir)
          total_index_j[k] = range(0, data_file_num, 1)
        total_index[j] = total_index_j
      gen_cp2k_task.gen_cp2k_task(cp2k_dic, work_dir, i, atoms_type_multi_sys, atoms_num_tot, total_index, \
                                  conv_new_data_num, choose_new_data_num_limit, train_stress, 'dp_test')
      if ( job_mode == 'workstation' ):
        cp2k_run.run_cp2kfrc_ws(work_dir, i, cp2k_exe, parallel_exe, cp2k_env_file, \
                                cp2k_job_per_node, proc_num_per_node, host, ssh, atoms_num_tot)
      elif ( job_mode == 'auto_submit' ):
        cp2k_run.run_cp2kfrc_as(work_dir, i, cp2k_queue, cp2k_exe, max_cp2k_job, \
                                cp2k_core_num, cp2k_env_file, submit_system, atoms_num_tot)

      write_data.write_restart_inp(inp_file, i, 4, data_num, work_dir)

    if ( restart_stage == 0 or restart_stage == 1 or restart_stage == 2 or restart_stage == 3 or restart_stage == 4 ):
      print ('Step 4: deep potential test', flush=True)
      sys_num, atoms_type_multi_sys, atoms_num_tot, use_bias_tot = \
      process.get_md_sys_info(lammps_dic, tot_atoms_type_dic)
      struct_index, success_ratio_sys, success_ratio  = \
      dp_test.active_learning_test(work_dir, i, atoms_type_multi_sys, use_bias_tot, \
                                   success_force_conv, max_force_conv, energy_conv)

      for j in range(len(success_ratio_sys)):
        print ('  The accurate ratio for system %d in iteration %d is %.2f%%' \
                %(j, i, success_ratio_sys[j]*100), flush=True)

      print ('  The accurate ratio for whole %d systems in iteration %d is %.2f%%' \
             %(sys_num, i, success_ratio*100), flush=True)

      if ( min(success_ratio_sys) >= 0.95 ):
        print (''.center(80,'*'), flush=True)
        print ('Cheers! deepff is converged!', flush=True)
        if ( i != 0 ):
          write_data.write_active_data(work_dir, i, tot_atoms_type_dic)
        exit()

      #Dump new data of cp2k
      for key in struct_index:
        task_num = len(struct_index[key])

        choosed_task = []
        choosed_index_num = []
        for j in range(task_num):
          choosed_index = struct_index[key][j]
          if ( len(choosed_index) < conv_new_data_num ):
            pass
          else:
            choosed_index_num.append(len(choosed_index))
            choosed_task.append(j)

        choosed_index_num_copy = copy.deepcopy(choosed_index_num)
        if ( sum(choosed_index_num)<choose_new_data_num_limit ):
          pass
        else:
          for j in range(len(choosed_index_num)):
            choosed_index_num[j]=int(choosed_index_num_copy[j]/sum(choosed_index_num_copy)*choose_new_data_num_limit)

        cp2k_sys_dir = ''.join((work_dir, '/iter_', str(i), '/03.cp2k_calc/sys_', str(key)))
        for j in range(len(choosed_task)):
          choosed_index = struct_index[key][choosed_task[j]]
          choosed_index_array = np.array(choosed_index)
          np.random.shuffle(choosed_index_array)
          choosed_index = list(choosed_index_array[0:choosed_index_num[j]])
          sys_task_index = data_op.comb_list_2_str(sorted(choosed_index), ' ')
          str_print = 'Choosed index for system %d cp2k task %d: %s' %(key, choosed_task[j], sys_task_index)
          str_print = data_op.str_wrap(str_print, 80, '  ')
          print (str_print, flush=True)

          cp2k_sys_task_dir = ''.join((cp2k_sys_dir, '/task_', str(choosed_task[j])))
          data_dir = ''.join((cp2k_sys_task_dir, '/data'))
          if ( not os.path.exists(data_dir) ):
            cmd = "mkdir %s" % ('data')
            call.call_simple_shell(cp2k_sys_task_dir, cmd)
          load_data.load_data_from_sepfile(cp2k_sys_task_dir, data_dir, 'traj_', 'cp2k', \
                                           tot_atoms_type_dic, sorted(choosed_index))
          energy_array, coord_array, frc_array, box_array, virial_array = load_data.read_raw_data(data_dir)
          train_data_num, test_data_num = load_data.raw_data_to_set(1, shuffle_data, data_dir, energy_array, \
                                                                    coord_array, frc_array, box_array, virial_array)
          if ( test_data_num > numb_test ):
            data_num.append(train_data_num)
          if ( test_data_num < numb_test and \
               success_ratio < float((active_learn_steps-train_data_num)/active_learn_steps) ):
            log_info.log_error('Warning: little selected structures, check the deepmd training.')
            exit()

      print ('  Success: dump new raw data of cp2k', flush=True)
      restart_stage = 0

    write_data.write_restart_inp(inp_file, i+1, 0, data_num, work_dir)

    if ( i == max_iter-1 ):
      log_info.log_error('Active learning does not converge')
      write_data.write_active_data(work_dir, i+1, tot_atoms_type_dic)

def kernel(work_dir, inp_file, deepff_type):

  '''
  kernel: kernel function to do active learning.

  Args:
    work_dir: string
      work_dir is the working directory of DPFlow.
    inp_file: string
      inp_file is the deepff input file.
    deepff_type: string
      deepff_type is the type of deepff.
  Return:
    none
  '''

  proc_num, proc_num_per_node, host, ssh = sys_info.get_host(work_dir)
  if ( len(data_op.list_replicate(proc_num_per_node)) != 1 ):
    host_str = data_op.comb_list_2_str(host, ' ')
    log_info.log_error('Resource error: the number of cores in %s are not equal, please submit your jobs to nodes with same number of cores' %(host_str))
    exit()

  if ( deepff_type == 'active_model_devi' ):
    deepff_key = ['deepmd_model', 'lammps', 'cp2k', 'active_learn', 'environ']
  elif ( deepff_type == 'active_dp_test' ):
    deepff_key = ['deepmd_test', 'lammps', 'cp2k', 'active_learn', 'environ']

  deepmd_dic, lammps_dic, cp2k_dic, active_learn_dic, environ_dic = \
  process.dump_input(work_dir, inp_file, deepff_key)

  if ( deepff_type == 'active_model_devi' ):
    deepff_key = ['deepmd_model', 'lammps', 'cp2k', 'active_learn', 'environ']
  elif ( deepff_type == 'active_dp_test' ):
    deepff_key = ['deepmd_test', 'lammps', 'cp2k', 'active_learn', 'environ']

  deepmd_dic, lammps_dic, cp2k_dic, active_learn_dic, environ_dic = \
  process.dump_input(work_dir, inp_file, deepff_key)

  environ_dic = check_deepff.check_environ(environ_dic, proc_num_per_node[0])
  dp_version = environ_dic['dp_version']
  if ( deepff_type == 'active_model_devi' ):
    deepmd_dic = check_deepff.check_deepmd_model(deepmd_dic, dp_version)
  elif ( deepff_type == 'active_dp_test' ):
    deepmd_dic = check_deepff.check_deepmd_test(deepmd_dic)
  active_learn_dic = check_deepff.check_active_learn(active_learn_dic)
  lammps_dic = check_deepff.check_lammps(lammps_dic, active_learn_dic)
  cp2k_dic = check_deepff.check_cp2k(cp2k_dic, lammps_dic)

  dp_path = sys_info.get_dp_path(work_dir)
  lmp_exe, lmp_path = sys_info.get_lmp_path(work_dir)
  mpi_path = sys_info.get_mpi_path(work_dir)
  cp2k_exe = environ_dic['cp2k_exe']

  print (data_op.str_wrap('DEEPFF| DEEPMD-KIT EXECUTABLE FILE IS %s' %(dp_path+'/dp'), 80), flush=True)
  print (data_op.str_wrap('DEEPFF| LAMMPS EXECUTABLE FILE IS %s' %(lmp_exe), 80), flush=True)
  if ( deepff_type == 'active_model_devi' ):
    print (data_op.str_wrap('DEEPFF| CP2K EXECUTABLE FILE IS %s' %(cp2k_exe), 80), flush=True)
  elif ( deepff_type == 'active_dp_test' ):
    print (data_op.str_wrap('DEEPFF| CP2K EXECUTABLE FILE IS %s\n' %(cp2k_exe), 80), flush=True)

  if ( deepff_type == 'active_model_devi' ):
    if ( 'set_data_dir' not in deepmd_dic['training'].keys() ):
      tot_atoms_type = process.get_atoms_type(deepmd_dic)
      if not operator.eq(deepmd_dic['model']['type_map'], tot_atoms_type) :
        type_map_str = data_op.comb_list_2_str(tot_atoms_type, ' ')
        log_info.log_error('Input error: type_map error, please use %s for deepff/deepmd/model/type_map and correspondingly change sel' %(type_map_str))
        exit()
    else:
      tot_atoms_type = deepmd_dic['model']['type_map']
    tot_atoms_type_dic = OrderedDict()
    for i in range(len(tot_atoms_type)):
      tot_atoms_type_dic[tot_atoms_type[i]] = i

    if ( len(deepmd_dic['model']['descriptor']['sel']) != len(tot_atoms_type) ):
      log_info.log_error('Input error: sel should be %d integers, please reset deepff/deepmd/model/descriptor/sel' %(len(tot_atoms_type)))
      exit()
    train_stress = deepmd_dic['training']['train_stress']
    restart_stage = active_learn_dic['restart_stage']
    restart_iter = active_learn_dic['restart_iter']
    init_train_data, init_data_num = process.dump_init_data(work_dir, deepmd_dic, train_stress, \
                                                            tot_atoms_type_dic, restart_iter, restart_stage)
    print ('DEEPFF| INITIAL TRAINING DATA:', flush=True)
    for i in range(len(init_train_data)):
      if ( i == len(init_train_data)-1 ):
        print ('%s\n' %(data_op.str_wrap(init_train_data[i], 80)), flush=True)
      else:
        print ('%s' %(data_op.str_wrap(init_train_data[i], 80)), flush=True)

    model_devi_iter(work_dir, inp_file, deepmd_dic, lammps_dic, cp2k_dic, active_learn_dic, \
                    environ_dic, init_train_data, init_data_num, tot_atoms_type_dic)
  elif ( deepff_type == 'active_dp_test' ):
    dp_test_iter(work_dir, inp_file, deepmd_dic, lammps_dic, active_learn_dic, cp2k_dic, environ_dic)

if __name__ == '__main__':

  from DPFlow.deepff import active
  work_dir = '/home/lujunbo/code/github/DPFlow/deepff/work_dir'
  inp_file = 'input.inp'
  max_cycle = 100
  active.kernel(work_dir, inp_file)
