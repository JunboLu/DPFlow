#!/usr/bin/env python

import os
import csv
import math
import linecache
import numpy as np
from DPFlow.tools import log_info
from DPFlow.tools import data_op
from DPFlow.tools import traj_info
from DPFlow.tools import call
from DPFlow.lib import geometry_mod
from DPFlow.analyze import center
from DPFlow.analyze import check_analyze

def distance(atoms_num, pre_base_block, end_base_block, pre_base, start_frame_id, \
             frames_num, each, init_step, end_step, atom_type_1, atom_type_2, \
             a_vec_tot, b_vec_tot, c_vec_tot, traj_coord_file, work_dir):

  '''
  distance: calculate distance between atom type 1 and atom type 2 over different frame.

  Args:
    atoms_num: int
      atoms_num is the number of atoms in the system.
    pre_base_block: int
      pre_base_block is the number of lines before structure in a structure block.
    end_base_block: int
      end_base_block is the number of lines after structure in a structure block.
    pre_base: int
      pre_base is the number of lines before block of the trajectory.
    start_frame_id: int
      start_frame_id is the starting frame id in the trajectory file.
    frames_num: int
      frames_num is the number of frames in the trajectory file.
    each: int
      each is printing frequency of md.
    init_step: int
      init_step is the initial step frame id.
    end_step: int
      end_step is the ending step frame id.
    atom_type_1: string
      atom_type_1 is the name of atom 1.
    atom_type_2: int
      atom_type_2 is the name of atom 2.
    a_vec_tot: 2-d float list, dim = n*3
      a_vec_tot is the cell vector a.
      Example: [[12.42, 0.0, 0.0],...,[12.42, 0.0, 0.0]]
    b_vec_tot: 2-d float list, dim = n*3
      b_vec_tot is the cell vector b.
      Example: [[0.0, 12.42, 0.0],...,[0.0, 12.42, 0.0]]
    c_vec_tot: 2-d float list, dim = n*3
      c_vec_tot is the cell vector c.
      Example: [[0.0, 0.0, 12.42],...,[0.0, 0.0, 12.42]]
    traj_coord_file: string
      file_name is the name of coordination trajectory file.
    work_dir: string
      work_dir is the working directory of DPFlow.
  Returns:
    distance: 3-d float list, dim = frames_num*(number of atom_1)*(number of atom_2)
              if atom_type_1 = atom_type_2, dim = frames_num*(number of atom_1 - 1)*(number of atom_2 - 1)
    atom_id_1: 1-d int list
      atom_id_1 contains atom id of atom_type_1.
    atom_id_2: 1-d int list
      atom_id_2 contains atom id of atom_type_2
  '''

  center_file = center.center(atoms_num, pre_base_block, end_base_block, pre_base, frames_num, \
                              a_vec_tot, b_vec_tot, c_vec_tot, 'center_box', 0, traj_coord_file, \
                              work_dir, 'center.xyz')

  atom_id_1 = []
  atom_id_2 = []
  for i in range(atoms_num):
    line_i = linecache.getline(center_file, pre_base_block+pre_base+i+1)
    line_i_split = data_op.split_str(line_i, ' ')
    if ( line_i_split[0] == atom_type_1 ):
      atom_id_1.append(i+1)
    if ( line_i_split[0] == atom_type_2 ):
      atom_id_2.append(i+1)

  frame_num_stat = int((end_step-init_step)/each+1)

  distance = []
  for i in range(frame_num_stat):
    distance_i = []
    id_label = int((init_step-start_frame_id)/each)+i
    a_vec = a_vec_tot[id_label]
    b_vec = b_vec_tot[id_label]
    c_vec = c_vec_tot[id_label]
    for j in range(atoms_num):
      line_j_num = id_label*(pre_base_block+atoms_num+end_base_block)+j+pre_base_block+pre_base+1
      line_j = linecache.getline(center_file, line_j_num)
      line_j_split = data_op.split_str(line_j, ' ', '\n')
      if ( line_j_split[0] == atom_type_1 ):
        coord_1 = []
        coord_2 = []
        for k in range(atoms_num):
          line_k_num = id_label*(pre_base_block+atoms_num+end_base_block)+k+pre_base_block+pre_base+1
          line_k = linecache.getline(center_file, line_k_num)
          line_k_split = data_op.split_str(line_k, ' ', '\n')
          if ( line_k_split[0] == atom_type_2 and j != k ):
            coord_1.append([float(line_j_split[1]),float(line_j_split[2]),float(line_j_split[3])])
            coord_2.append([float(line_k_split[1]),float(line_k_split[2]),float(line_k_split[3])])
        dist = geometry_mod.geometry.calculate_distance(np.asfortranarray(coord_1, dtype='float32'), \
                                                        np.asfortranarray(coord_2, dtype='float32'), \
                                                        np.asfortranarray(a_vec, dtype='float32'), \
                                                        np.asfortranarray(b_vec, dtype='float32'), \
                                                        np.asfortranarray(c_vec, dtype='float32'))
        distance_i.append(list(dist))
    distance.append(distance_i)

  linecache.clearcache()

  cmd = 'rm %s' %(center_file)
  call.call_simple_shell(work_dir, cmd)

  return distance, atom_id_1, atom_id_2

def rdf(distance, a_vec_tot, b_vec_tot, c_vec_tot, r_increment, work_dir):

  '''
  rdf: get rdf between atom type 1 and atom type 2

  Args:
    distance: 3-d float list, dim = frames_num*(number of atom_1)*(number of atom_2)
    a_vec_tot: 2-d float list, dim = n*3
      a_vec_tot is the cell vector a.
      Example: [[12.42, 0.0, 0.0],...,[12.42, 0.0, 0.0]]
    b_vec_tot: 2-d float list, dim = n*3
      b_vec_tot is the cell vector b.
      Example: [[0.0, 12.42, 0.0],...,[0.0, 12.42, 0.0]]
    c_vec_tot: 2-d float list, dim = n*3
      c_vec_tot is the cell vector c.
      Example: [[0.0, 0.0, 12.42],...,[0.0, 0.0, 12.42]]
    r_increment: float
      r_increment is the increment of r.
    work_dir: string
      work_dir is the working directory of DPFlow.
  Returns:
    rdf_file: string
      rdf_file contains rdf information.
  '''

  vec = a_vec_tot[len(a_vec_tot)-1]+b_vec_tot[len(a_vec_tot)-1]+c_vec_tot[len(a_vec_tot)-1]
  r_max = np.sqrt(np.dot(vec, vec))/2.0
  #vol = 4.0/3.0*np.pi*r_max**3
  vol = []
  for i in range(len(a_vec_tot)):
    vol.append(np.dot(a_vec_tot[i], np.cross(b_vec_tot[i], c_vec_tot[i])))
  data_num = int(r_max/r_increment)

  rdf_value, integral_value = \
  geometry_mod.geometry.rdf(distance, np.asfortranarray(vol, dtype='float32'), r_increment, data_num)

  rdf_file = ''.join((work_dir, '/rdf_integral.csv'))
  with open(rdf_file ,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['distance(Ang)', 'rdf', 'int'])
    for i in range(data_num-1):
      writer.writerow([r_increment*(i+1), rdf_value[i], integral_value[i]])

  return rdf_file

def rdf_run(rdf_param, work_dir):

  '''
  rdf_run: the kernel function to run rdf function. It will call rdf.

  Args:
    rdf_param: dictionary
      rdf_param contains keywords used in rdf functions.
    work_dir: string
      work_dir is the working directory of DPFlow.
  Returns:
    none
  '''

  rdf_param = check_analyze.check_rdf_inp(rdf_param)
  init_step = rdf_param['init_step']
  end_step = rdf_param['end_step']
  r_increment = rdf_param['r_increment']

  traj_coord_file = rdf_param['traj_coord_file']
  atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_frame_id, end_frame_id, time_step = \
  traj_info.get_traj_info(traj_coord_file, 'coord_xyz')

  log_info.log_traj_info(atoms_num, frames_num, each, start_frame_id, end_frame_id, time_step)

  atom_type_pair = rdf_param['atom_type_pair']
  atom_1 = atom_type_pair[0]
  atom_2 = atom_type_pair[1]

  atoms = []
  for i in range(atoms_num):
    line_i = linecache.getline(traj_coord_file, pre_base_block+pre_base+i+1)
    line_i_split = data_op.split_str(line_i, ' ', '\n')
    atoms.append(line_i_split[0])
  atom_type = data_op.list_replicate(atoms)

  if atom_1 not in atom_type:
    log_info.log_error('Input error: %s atom type is not in the system' %(atom_1))
    exit()

  if atom_2 not in atom_type:
    log_info.log_error('Input error: %s atom type is not in the system' %(atom_2))
    exit()

  md_type = rdf_param['md_type']
  if ( md_type == 'nvt' or md_type == 'nve' ):
    a_vec = rdf_param['box']['A']
    b_vec = rdf_param['box']['B']
    c_vec = rdf_param['box']['C']
    a_vec_tot = []
    b_vec_tot = []
    c_vec_tot = []
    for i in range(frames_num):
      a_vec_tot.append(a_vec)
      b_vec_tot.append(b_vec)
      c_vec_tot.append(c_vec)
  elif ( md_type == 'npt' ):
    traj_cell_file = rdf_param['traj_cell_file']
    a_vec_tot = []
    b_vec_tot = []
    c_vec_tot = []
    for i in range(frames_num):
      id_label = int((init_step-start_frame_id)/each)+i
      line_i = linecache.getline(traj_cell_file, i+2)
      line_i_split = data_op.split_str(line_i, ' ', '\n')
      a_vec_tot.append([float(line_i_split[2]), float(line_i_split[3]), float(line_i_split[4])])
      b_vec_tot.append([float(line_i_split[5]), float(line_i_split[6]), float(line_i_split[7])])
      c_vec_tot.append([float(line_i_split[8]), float(line_i_split[9]), float(line_i_split[10])])

    linecache.clearcache()

  print ('RDF'.center(80, '*'), flush=True)
  print ('Analyze radial distribution function between %s and %s' %(atom_1, atom_2), flush=True)
  dist, atom_id_1, atom_id_2 = distance(atoms_num, pre_base_block, end_base_block, pre_base, start_frame_id, \
                                        frames_num, each, init_step, end_step, atom_1, atom_2, a_vec_tot, \
                                        b_vec_tot, c_vec_tot, traj_coord_file, work_dir)

  rdf_file = rdf(dist, a_vec_tot[int(init_step/each):int(end_step/each)+1], \
                 b_vec_tot[int(init_step/each):int(end_step/each)+1], \
                 c_vec_tot[int(init_step/each):int(end_step/each)+1], r_increment, work_dir)

  str_print = 'The rdf file is written in %s' %(rdf_file)
  print (data_op.str_wrap(str_print, 80), flush=True)
