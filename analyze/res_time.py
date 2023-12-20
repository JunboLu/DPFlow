#! /env/bin/python

import math
import linecache
import numpy as np
from DPFlow.tools import log_info
from DPFlow.tools import data_op
from DPFlow.tools import traj_info
from DPFlow.analyze import rdf
from DPFlow.analyze import check_analyze

def generate_mask(distance, first_shell_dist, dist_conv, work_dir):

  '''
  generate_mask: generate the mask of atom 2

  Args:
    distance: 3-d float list, dim = frames_num*(number of atom_1)*(number of atom_2)
    first_shell_dist: float
      dist_first_shell is the distance of the first shell.
    dist_conv: float
      dist_conv is the converge value for the first_shell_dist.
    work_dir: string
      work_dir is working directory of DPFlow.
  Returns:
    mask_file: string
      mask_file is the mask file.
  '''

  dim1, dim2, dim3 = np.array(distance).shape
  mask_file_name = ''.join((work_dir, '/coord.mask'))
  mask_file = open(mask_file_name, 'w')
  for i in range(dim1):
    mask = []
    for k in range(dim3):
      mask_ik = '0'
      for j in range(dim2):
        if ( abs(distance[i][j][k] - first_shell_dist) < dist_conv \
             or distance[i][j][k] < first_shell_dist ):
          mask_ik = '1'
          break
      mask.append(mask_ik)
    mask_line = ''
    for k in range(dim3):
      mask_line = mask_line + ' ' + mask[k]
    mask_file.write('%d %s\n' %(i, mask_line))

  mask_file.close()

  return mask_file_name

def res_time_run(res_time_param, work_dir):

  '''
  res_time_run: the kernel function to run res_time function.

  Args:
    res_time_param: dictionary
      res_time_param contains keywords used to analyze residence time.
    work_dir: string
      work_dir is the working directory of DPFlow.
  Returns:
    none
  '''

  res_time_param = check_analyze.check_res_time_inp(res_time_param)

  md_type = res_time_param['md_type']
  traj_coord_file = res_time_param['traj_coord_file']

  atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_frame_id, end_frame_id, time_step = \
  traj_info.get_traj_info(traj_coord_file, 'coord_xyz')

  log_info.log_traj_info(atoms_num, frames_num, each, start_frame_id, end_frame_id, time_step)

  atom_type_pair = res_time_param['atom_type_pair']
  atom_type_1 = atom_type_pair[0]
  atom_type_2 = atom_type_pair[1]

  first_shell_dist = res_time_param['first_shell_dist']
  dist_conv = res_time_param['dist_conv']
  init_step = res_time_param['init_step']
  end_step = res_time_param['end_step']

  if ( md_type == 'nvt' or md_type == 'nve' ):
    a_vec = res_time_param['box']['A']
    b_vec = res_time_param['box']['B']
    c_vec = res_time_param['box']['C']
    a_vec_tot = []
    b_vec_tot = []
    c_vec_tot = []
    for i in range(frames_num):
      a_vec_tot.append(a_vec)
      b_vec_tot.append(b_vec)
      c_vec_tot.append(c_vec)
  elif ( md_type == 'npt' ):
    traj_cell_file = res_time_param['traj_cell_file']
    a_vec_tot = []
    b_vec_tot = []
    c_vec_tot = []
    for i in range(frames_num):
      line_i = linecache.getline(traj_cell_file, i+2)
      line_i_split = data_op.split_str(line_i, ' ', '\n')
      a_vec_tot.append([float(line_i_split[2]), float(line_i_split[3]), float(line_i_split[4])])
      b_vec_tot.append([float(line_i_split[5]), float(line_i_split[6]), float(line_i_split[7])])
      c_vec_tot.append([float(line_i_split[8]), float(line_i_split[9]), float(line_i_split[10])])

    linecache.clearcache()

  print ('RESIDENCE-TIME'.center(80, '*'), flush=True)
  print ('Analyze residence time of %s' %(atom_type_2), flush=True)

  #distance is 3-d float list (frames_num*(number of atom_1)*(number of atom_2)).
  distance, atom_1, atom_2 = rdf.distance(atoms_num, pre_base_block, end_base_block, pre_base, start_frame_id, \
                                          frames_num, each, init_step, end_step, atom_type_1, atom_type_2, \
                                          a_vec_tot, b_vec_tot, c_vec_tot, traj_coord_file, work_dir)

  mask_file = generate_mask(distance, first_shell_dist, dist_conv, work_dir)

  frames_num = len(open(mask_file).readlines())
  dim1, dim2, atoms_2_num = np.array(distance).shape
  mask_mat = np.zeros([frames_num, atoms_2_num])

  for i in range(frames_num):
    line = linecache.getline(mask_file, i+1)
    line_split = data_op.split_str(line, ' ', '\n')
    for j in range(atoms_2_num):
      mask_mat[i,j] = int(line_split[j+1])

  T = np.zeros(atoms_2_num)
  Tnum = np.zeros(atoms_2_num)
  Ttot = np.zeros(atoms_2_num)
 
  mask_mat_sum = np.sum(mask_mat, axis=0)
  T_stat = []
  Tnum_stat = [] 
  for i in range(atoms_2_num):
    Ttot[i] = mask_mat_sum[i]
    if ( Ttot[i] != 0 ):
      for j in range(frames_num-1):
        if ( mask_mat[j+1,i] != mask_mat[j,i] ):
          Tnum[i] = Tnum[i]+0.5
      if ( Tnum[i] > 0 ):
        print (Ttot[i]*each*time_step, Tnum[i])
        T_stat.append(Ttot[i]/math.ceil(Tnum[i]))

  residence_time = sum(T_stat)*each*time_step/len(T_stat)
  #np.sum(T)*each*time_step*np.sum(Ttot > 0)

  print ('The residence time is %f fs' %(residence_time), flush=True)
