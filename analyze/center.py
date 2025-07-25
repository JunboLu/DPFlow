#!/usr/bin/env python

import os
import linecache
import numpy as np
from DPFlow.tools import atom
from DPFlow.tools import log_info
from DPFlow.tools import traj_info
from DPFlow.tools import data_op
from DPFlow.lib import geometry_mod
from DPFlow.analyze import geometry
from DPFlow.analyze import check_analyze

def center(atoms_num, pre_base_block, end_base_block, pre_base, frames_num, a_vec_tot, b_vec_tot, \
           c_vec_tot, center_type, center_id, traj_coord_file, work_dir, file_name, trans_type=1, \
           group_atom_1_id=[[]], group_atoms_mass=[[]]):

  #Reference literature: Computer simulation of liquids, Oxford University press.

  '''
  center: center the system in trajectory file.

  Args:
    atoms_num: int
      atoms_num is the number of atoms in the system.
    pre_base_block: int
      pre_base_block is the number of lines before structure in a structure block.
    end_base_block: int
      end_base_block is the number of lines after structure in a structure block.
    pre_base: int
      pre_base is the number of lines before block of trajectory.
    frames_num: int
      frames_num is the number of frames in trajectory file.
    a_vec_tot: 2-d float list, dim = n*3
      a_vec_tot is the cell vector a.
      Example: [[12.42, 0.0, 0.0],...,[12.42, 0.0, 0.0]]
    b_vec_tot: 2-d float list, dim = n*3
      b_vec_tot is the cell vector b.
      Example: [[0.0, 12.42, 0.0],...,[0.0, 12.42, 0.0]]
    c_vec_tot: 2-d float list, dim = n*3
      c_vec_tot is the cell vector c.
      Example: [[0.0, 0.0, 12.42],...,[0.0, 0.0, 12.42]]
    center_type : string
      center_type is the type of center. Two choices : center_box, center_image.
    center_id: int
      center_id is the id of center atom. If we use center_box, center_id is 0.
      If we use center_image, we must set it.
    traj_coord_file: string
      traj_coord_file is the name of coordination trajectory file.
    work_dir: string
      work_dir is working directory of DPFlow.
    file_name: string
      file_name is the name of generated file.
    trans_type: int
      trans_type is the translation type when doing center.
      If trans_type is 0, we will consider the connectivity of molecules in the group.
      If trans_type is 1, we will never consider the connectivity of molecules in the group.
    group_atom_1_id: 2-d int list
      group_atom_1_id is the id of first atoms in the molecules in the group.
    group_atoms_mass: 2-d float list
      group_atoms_mass contains the atoms mass for each group.
  Return:
    center_file_name: string
      center_file_name is the name of centered file.
  '''

  #In general, the center function will not consider connectivity.

  mass_list_len = []
  id_list_len = []
  for i in range(len(group_atoms_mass)):
    mass_list_len.append(len(group_atoms_mass[i]))
    id_list_len.append(len(group_atom_1_id[i]))

  group_atom_1_id = data_op.expand_2d_list(group_atom_1_id, max(id_list_len), 0)
  group_atoms_mass = data_op.expand_2d_list(group_atoms_mass, max(mass_list_len), 0)

  center_file_name = ''.join((work_dir, '/', file_name))
  center_file = open(center_file_name, 'w')

  for i in range(frames_num):
    a_vec = a_vec_tot[i]
    b_vec = b_vec_tot[i]
    c_vec = c_vec_tot[i]
    #Dump atoms, atoms_mass and coordinates from trajectory file.
    coord = np.asfortranarray(np.zeros((atoms_num, 3)),dtype='float32')
    line_1 = linecache.getline(traj_coord_file, i*(pre_base_block+atoms_num+end_base_block)+1)
    line_2 = linecache.getline(traj_coord_file, i*(pre_base_block+atoms_num+end_base_block)+2)
    atoms = []
    atoms_mass = []
    for j in range(atoms_num):
      line_ij = linecache.getline(traj_coord_file, i*(pre_base_block+atoms_num+end_base_block)+j+1+pre_base_block+pre_base)
      line_ij_split = data_op.split_str(line_ij, ' ', '\n')
      atoms.append(line_ij_split[0])
      atoms_mass.append(atom.get_atom_mass(line_ij_split[0])[1])
      coord[j,0] = float(line_ij_split[1])
      coord[j,1] = float(line_ij_split[2])
      coord[j,2] = float(line_ij_split[3])

    if ( center_type == "center_box" ):
      new_coord = geometry_mod.geometry.periodic_center_box(coord, np.asfortranarray(a_vec, dtype='float32'), \
                                                            np.asfortranarray(b_vec, dtype='float32'), \
                                                            np.asfortranarray(c_vec, dtype='float32'), \
                                                            trans_type, np.asfortranarray(group_atom_1_id, dtype='int32'), \
                                                            np.asfortranarray(group_atoms_mass, dtype='float32'), \
                                                            np.asfortranarray(id_list_len, dtype='int32'), \
                                                            np.asfortranarray(mass_list_len, dtype='int32'))
    elif ( center_type == "center_image" ):
      #We translate the atoms in the box at first, and then image
      new_coord_box = geometry_mod.geometry.periodic_center_box(coord, np.asfortranarray(a_vec, dtype='float32'), \
                                                                np.asfortranarray(b_vec, dtype='float32'), \
                                                                np.asfortranarray(c_vec, dtype='float32'), \
                                                                trans_type, np.asfortranarray(group_atom_1_id, dtype='int32'), \
                                                                np.asfortranarray(group_atoms_mass, dtype='float32'), \
                                                                np.asfortranarray(id_list_len, dtype='int32'), \
                                                                np.asfortranarray(mass_list_len, dtype='int32'))

      center_coord = np.asfortranarray(new_coord_box[center_id-1], dtype='float32')

      new_coord = geometry_mod.geometry.periodic_center_image(new_coord_box, np.asfortranarray(a_vec, dtype='float32'), \
                                                              np.asfortranarray(b_vec, dtype='float32'), \
                                                              np.asfortranarray(c_vec, dtype='float32'), center_coord, \
                                                              trans_type, np.asfortranarray(group_atom_1_id, dtype='int32'), \
                                                              np.asfortranarray(group_atoms_mass, dtype='float32'), \
                                                              np.asfortranarray(id_list_len, dtype='int32'), \
                                                              np.asfortranarray(mass_list_len, dtype='int32'))

    new_coord = geometry_mod.geometry.trans_box_center(new_coord, np.asfortranarray(atoms_mass,dtype='float32'), \
                                                       np.asfortranarray(a_vec, dtype='float32'), \
                                                       np.asfortranarray(b_vec, dtype='float32'), \
                                                       np.asfortranarray(c_vec, dtype='float32'))

    center_file.write(line_1)
    center_file.write(line_2)

    for j in range(atoms_num):
      center_file.write('%3s%21.10f%20.10f%20.10f\n' \
                        %(atoms[j], new_coord[j,0], new_coord[j,1], new_coord[j,2]))

  linecache.clearcache()
  center_file.close()

  return center_file_name

def center_run(center_param, work_dir):

  '''
  center_run: kernel function to run center. It will call center function.

  Args:
    center_param: dictionary
      center_param contains keywords used in center function.
    work_dir: string
      work_dir is the working directory of DPFlow.
  Returns :
    none
  '''

  center_param = check_analyze.check_center_inp(center_param)
  center_type = center_param['center_type']
  traj_coord_file = center_param['traj_coord_file']

  if ( center_type == 'center_image' ):
    center_id = center_param['center_atom_id']
  else:
    center_id = 0

  atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_frame_id, end_frame_id, time_step = \
  traj_info.get_traj_info(traj_coord_file, 'coord_xyz')

  log_info.log_traj_info(atoms_num, frames_num, each, start_frame_id, end_frame_id, time_step)

  md_type = center_param['md_type']
  if ( md_type == 'nvt' or md_type == 'nve' ):
    a_vec = center_param['box']['A']
    b_vec = center_param['box']['B']
    c_vec = center_param['box']['C']
    a_vec_tot = []
    b_vec_tot = []
    c_vec_tot = []
    for i in range(frames_num):
      a_vec_tot.append(a_vec)
      b_vec_tot.append(b_vec)
      c_vec_tot.append(c_vec)
  elif ( md_type == 'npt' ):
    traj_cell_file = center_param['traj_cell_file']
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

  print ('CENTER'.center(80,'*'), flush=True)

  if 'connect0' in center_param.keys():
    atom_id = center_param['connect0']['atom_id']
    group_atom = center_param['connect0']['group_atom']

    atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, \
    start_frame_id, end_frame_id, time_step, group_atom_1_id, group_atoms_mass = \
    traj_info.get_traj_info(traj_coord_file, 'coord_xyz', group_atom, atom_id, True)

    center_file = center(atoms_num, pre_base_block, end_base_block, pre_base, frames_num, \
                         a_vec_tot, b_vec_tot, c_vec_tot, center_type, center_id, traj_coord_file, \
                         work_dir, 'center.xyz', 0, group_atom_1_id, group_atoms_mass)
  else:
    center_file = center(atoms_num, pre_base_block, end_base_block, pre_base, frames_num, a_vec_tot, \
                         b_vec_tot, c_vec_tot, center_type, center_id, traj_coord_file, work_dir, 'center.xyz')

  print (data_op.str_wrap('The centered trajectory is written in %s' %(center_file), 80), flush=True)

if __name__ == '__main__':
  from DPFlow.tools import traj_info
  from DPFlow.analyze import center
  traj_coord_file = 'test.xyz'
  groups = [['Mn','F','O','O','O']]
  atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_frame_id, end_frame_id, \
  time_step, group_atom_1_id, group_atoms_num = \
  traj_info.get_traj_info(file_name, groups, True)

  boxl = 18.898
  center.center(atoms_num, pre_base_block, end_base_block, pre_base, frames_num, boxl, \
                group_atom_1_id, group_atoms_num, traj_coord_file, work_dir, 'center_box', 0)

