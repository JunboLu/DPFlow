#!/usr/bin/env python

import os
import linecache
from DPFlow.tools import traj_tools
from DPFlow.tools import traj_info
from DPFlow.tools import data_op
from DPFlow.tools import log_info
from DPFlow.tools import file_tools
from DPFlow.tools import revise_cp2k_inp

mulliken_pop_file_pre_num = 5
mulliken_pop_file_late_num = 3

def restart_history(restart_id, proj_name, QMMM, QM_num, data_dir):

  pos_file = ''.join((data_dir, '/', proj_name, '-pos-1.xyz'))
  if os.path.exists(pos_file):
    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(pos_file, 'coord_xyz')
    pos_store_line_num = (int((restart_id-start_id)/each)+1)*(pre_base_block+atoms_num+end_base_block)+pre_base+1
    pos_whole_line_num = len(open(pos_file).readlines())
    os.environ["var_1"] = str(pos_store_line_num)
    os.environ["var_2"] = str(pos_whole_line_num)
    os.environ["var_3"] = pos_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(pos_store_line_num, pos_whole_line_num, pos_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  vel_file = ''.join((data_dir, '/', proj_name, '-vel-1.xyz'))
  if os.path.exists(vel_file):
    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(vel_file, 'vel')
    vel_store_line_num = (int((restart_id-start_id)/each)+1)*(pre_base_block+atoms_num+end_base_block)+pre_base+1
    vel_whole_line_num = len(open(vel_file).readlines())
    os.environ["var_1"] = str(vel_store_line_num)
    os.environ["var_2"] = str(vel_whole_line_num)
    os.environ["var_3"] = vel_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(vel_store_line_num, vel_whole_line_num, vel_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  frc_file = ''.join((data_dir, '/', proj_name, '-frc-1.xyz'))
  if os.path.exists(frc_file):
    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(frc_file, 'frc')
    frc_store_line_num = (int((restart_id-start_id)/each)+1)*(pre_base_block+atoms_num+end_base_block)+pre_base+1
    frc_whole_line_num = len(open(frc_file).readlines())
    os.environ["var_1"] = str(frc_store_line_num)
    os.environ["var_2"] = str(frc_whole_line_num)
    os.environ["var_3"] = frc_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(frc_store_line_num, frc_whole_line_num, frc_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  ene_file = ''.join((data_dir, '/', proj_name, '-1.ener'))
  if os.path.exists(ene_file):
    blocks_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(ene_file, 'ener')
    ene_store_line_num = (int((restart_id-start_id)/each)+1)+pre_base+1
    ene_whole_line_num = len(open(ene_file).readlines())
    os.environ["var_1"] = str(ene_store_line_num)
    os.environ["var_2"] = str(ene_whole_line_num)
    os.environ["var_3"] = ene_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(ene_store_line_num, ene_whole_line_num, ene_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  cell_file = ''.join((data_dir, '/', proj_name, '-1.cell'))
  if os.path.exists(cell_file):
    blocks_num, pre_base_block, cell_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(cell_file, 'cell')
    cell_store_line_num = (int((restart_id-start_id)/each)+1)+pre_base+1
    cell_whole_line_num = len(open(cell_file).readlines())
    os.environ["var_1"] = str(cell_store_line_num)
    os.environ["var_2"] = str(cell_whole_line_num)
    os.environ["var_3"] = cell_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(cell_store_line_num, cell_whole_line_num, cell_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  mix_ene_file = ''.join((data_dir,'/',  proj_name, '-mix-1.ener'))
  if os.path.exists(mix_ene_file):
    blocks_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(mix_ene_file, 'mix_ener')
    ene_store_line_num = (int((restart_id-start_id)/each)+1)+pre_base+1
    ene_whole_line_num = len(open(mix_ene_file).readlines())
    os.environ["var_1"] = str(ene_store_line_num)
    os.environ["var_2"] = str(ene_whole_line_num)
    os.environ["var_3"] = mix_ene_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(ene_store_line_num, ene_whole_line_num, mix_ene_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  pop_file = ''.join((data_dir, '/', proj_name, '-1.mulliken'))
  pos_file = ''.join((data_dir, '/', proj_name, '-pos-1.xyz'))
  if os.path.exists(pop_file) and os.path.exists(pos_file):
    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(pos_file, 'coord_xyz')
    if QMMM:
      pop_store_line_num = (int((restart_id-start_id)/each)+1)*(mulliken_pop_file_pre_num+mulliken_pop_file_late_num+atoms_num)+1
      pop_whole_line_num = len(open(pop_file).readlines())
      os.environ["var_1"] = str(pop_store_line_num)
      os.environ["var_2"] = str(pop_whole_line_num)
      os.environ["var_3"] = pop_file
      os.system("sed -ie ''$var_1','$var_2'd' $var_3")
    else:
      pop_store_line_num = (restart_id-start_id+1)*(mulliken_pop_file_pre_num+mulliken_pop_file_late_num+QM_num)+1
      pop_whole_line_num = len(open(pop_file).readlines())
      os.environ["var_1"] = str(pop_store_line_num)
      os.environ["var_2"] = str(pop_whole_line_num)
      os.environ["var_3"] = pop_file
      os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(pop_store_line_num, pop_whole_line_num, pop_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

  lagrange_file = ''.join((data_dir, '/', proj_name, '-1.LagrangeMultLog'))
  pos_file = ''.join((data_dir, '/', proj_name, '-pos-1.xyz'))
  if os.path.exists(lagrange_file) and os.path.exists(pos_file):
    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_id, end_id, time_step = \
    traj_info.get_traj_info(pos_file, 'coord_xyz')
    blocks_num, pre_base_lag, pre_base_block_lag, end_base_block_lag, frame_start = traj_tools.get_block_base(lagrange_file)
    lag_store_line_num = (int((restart_id-start_id)/each)+1)*(pre_base_block_lag+blocks_num+end_base_block_lag)+1
    lag_whole_line_num = len(open(lagrange_file).readlines())
    print (lag_store_line_num,lag_whole_line_num)
    os.environ["var_1"] = str(lag_store_line_num)
    os.environ["var_2"] = str(lag_whole_line_num)
    os.environ["var_3"] = lagrange_file
    os.system("sed -ie ''$var_1','$var_2'd' $var_3")

    str_print = "Success: delete lines from No.%d to No.%d for %s file" %(lag_store_line_num, lag_whole_line_num, lagrange_file)
    str_print = data_op.str_wrap(str_print, 80, '')
    print (str_print, flush=True)

def handle_restart_run(handle_restart_param):

  if ( 'data_dir' in handle_restart_param.keys() ):
    data_dir = handle_restart_param['data_dir']
  else:
    log_info.log_error('No data directory found, please set data_dir')
    exit()

  if ( 'cp2k_inp_file' in handle_restart_param.keys() ):
    cp2k_inp_file = handle_restart_param['cp2k_inp_file']
    if ( os.path.exists(os.path.abspath(os.path.expanduser(cp2k_inp_file))) ):
      cp2k_inp_file = os.path.abspath(os.path.expanduser(cp2k_inp_file))
      proj_name = revise_cp2k_inp.get_proj_name(cp2k_inp_file, data_dir)
    else:
      log_info.log_error('cp2k input file %s does not exist, please set cp2k_inp_file correctly' %(cp2k_inp_file))
      exit()
  else:
    log_info.log_error('No cp2k input file found, please set cp2k_inp_file')
    exit()

  restart_file = ''.join((proj_name, '-1.restart'))

  start_step_line_num = file_tools.grep_line_num('STEP_START_VAL', restart_file, data_dir)
  start_step_line = linecache.getline(restart_file, start_step_line_num[0])
  linecache.clearcache()
  start_step_line_split = data_op.split_str(start_step_line, ' ', '\n')
  restart_step = int(start_step_line_split[1])

  if ( 'QMMM' in handle_restart_param.keys() ):
    QMMM = int(handle_restart_param['QMMM'])
  else:
    QMMM = 0 # 0 means no QMMM method

  if ( 'QM_num' in handle_restart_param.keys() ):
    QM_num = handle_restart_param['QM_num']
  else:
    QM_num = 0

  print ('HANDLE_RESTART'.center(80, '*'), flush=True)
  restart_history(restart_step, proj_name, QMMM, QM_num, data_dir)

