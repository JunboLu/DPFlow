#! /use/env/bin python

import os
import linecache
import numpy as np
from collections import OrderedDict
from DPFlow.tools import *
from DPFlow.lib.geometry_mod import geometry

def choose_lmp_str(work_dir, iter_id, atoms_type_multi_sys, use_bias_tot, success_force_conv, max_force_conv):

  '''
  choose_lmp_str: choose lammps structure based on force-force correlation.

  Args:
    work_dir: string
      work_dir is the workding directory of DPFlow.
    iter_id: int
      iter_id is current iteration number.
    atoms_type_multi_sys: 2-d dictionary, dim = (num of lammps systems) * (num of atom types)
      atoms_type_multi_sys is the atoms type for multi-systems.
      example: {0:{'O':1,'H':2,'N':3},1:{'O':1,'S':2,'N':3}}
    success_force_conv: float
      success_force_conv is the maximum force convergence.
  Returns:
    struct_index: dictionary
      example: {0:{0:[2,4,6...], 1:[2,3,4...]}, 1:{0:[3,4,6...], 1:[5,6,7...]}}
                 !                !                    !
               sys_id          task_id              traj_id
    success_ratio_sys: 1-d float list
      success_ratio is the successful ratio for different systems.
    success_ratio: float
      success_ratio is the successful ratio for whole systems.
    success_devi_ratio: float
      success_devi_conv is the deviation successful ratio for whole systems.
  '''

  struct_index = OrderedDict()
  success_frames = []
  success_devi_frames = []
  tot_frames = []
  lammps_dir = ''.join((work_dir, '/iter_', str(iter_id), '/02.lammps_calc'))
  sys_num = len(atoms_type_multi_sys)

  for i in range(sys_num):
    struct_index_i = OrderedDict()
    lammps_sys_dir = ''.join((lammps_dir, '/sys_', str(i)))

    cmd = "ls | grep %s" % ("'task_[0-9]'")
    task_num = len(call.call_returns_shell(lammps_sys_dir, cmd))
    tot_frames_i = []
    success_frames_i = []
    success_devi_frames_i = []
    for j in range(task_num):
      lammps_sys_task_dir = ''.join((lammps_sys_dir, '/task_', str(j)))
      data_dir = ''.join((lammps_sys_task_dir, '/data'))

      cmd = "ls | grep %s" % ("'model_[0-9]'")
      model_num = len(call.call_returns_shell(lammps_sys_task_dir, cmd))

      log_file = ''.join((lammps_sys_task_dir, '/lammps.out'))
      dump_file = ''.join((lammps_sys_task_dir, '/atom.dump'))
      atoms_num, frames_num, start_id, end_id, each = read_lmp.lmp_traj_info(dump_file, log_file)
      atoms, energy, coord, vel, frc, cell = \
      read_lmp.read_lmp_log_traj(dump_file, log_file, atoms_type_multi_sys[i], [], True, True, False, True, True)
      tot_frames_i.append(frames_num)

      model_devi_file_name_abs = ''.join((lammps_sys_task_dir, '/model_devi.out'))
      model_devi_file = open(model_devi_file_name_abs, 'w')
      model_devi_file.write('Frame     MAX_E(eV)     MIN_E(eV)     AVG_E(eV)     MAX_F(eV/A)     MIN_F(eV/A)     AVG_F(eV/A)\n')

      choosed_index = []

      #Get box parameter, we will calculate distance later.
      success_frames_ij = 0
      success_devi_frames_ij = 0

      for k in range(frames_num):
        ene_model = []
        frc_model = []
        #Dump energies and forces for models.
        for l in range(model_num):
          model_dir = ''.join((lammps_sys_task_dir, '/model_', str(l)))
          if ( l == 0 and not use_bias_tot[i] ):
            ene_model.append(energy[k])
          else:
            model_log_file = ''.join((model_dir, '/traj_', str(k), '/lammps.out'))
            step_line_num = file_tools.grep_line_num('Step ', model_log_file, lammps_sys_task_dir)[0]
            line = linecache.getline(model_log_file, step_line_num+1)
            line_split = data_op.split_str(line, ' ', '\n')
            ene_model.append(float(line_split[2]))
            linecache.clearcache()
          id_l = []
          frc_model_l = []
          model_dump_file = ''.join((model_dir, '/traj_', str(k), '/atom.dump'))
          for m in range(atoms_num):
            line_lm = linecache.getline(model_dump_file, m+9+1)
            line_lm_split = data_op.split_str(line_lm, ' ', '\n')
            id_l.append(int(line_lm_split[0]))
            frc_model_l.append([float(line_lm_split[5]), \
                                float(line_lm_split[6]), \
                                float(line_lm_split[7])])

          linecache.clearcache()
          id_l_asc, asc_index = data_op.get_list_order(id_l, 'ascend', True)
          frc_model_l_asc = data_op.reorder_list(frc_model_l, asc_index)
          frc_model.append(frc_model_l_asc)

        ene_devi = []
        #frc_devi = []
        for l in range(len(frc_model)):
          for m in range(len(frc_model)):
            if ( l < m ):
              ene_devi.append(abs(ene_model[l]-ene_model[m]))
              #frc_devi.append(calc_force_devi(frc_model[l], frc_model[m]))

        mean_frc = np.mean(np.array(frc_model), axis=0) #(N, 3)
        deviations = np.array(frc_model) - mean_frc     #(M, N, 3)
        squared_norms = np.sum(deviations**2, axis=2)   #(M, N)
        sigma = np.sqrt(np.mean(squared_norms, axis=0)) #(N)

        #frc_devi_avg = []
        #for l in range(atoms_num):
        #  sum_value = 0.0
        #  for m in range(len(frc_devi)):
        #    sum_value = sum_value + frc_devi[m][l]
        #  frc_devi_avg.append(sum_value/len(frc_devi))

        max_ene = max(ene_devi)
        min_ene = min(ene_devi)
        avg_ene = sum(ene_devi)/len(ene_devi)
        max_frc = np.max(sigma)
        min_frc = np.min(sigma)
        avg_frc = np.mean(sigma)
        #max_frc = max(frc_devi_avg)
        #min_frc = min(frc_devi_avg)
        #avg_frc = sum(frc_devi_avg)/len(frc_devi_avg)
        model_devi_file.write('%-10d%-14.6f%-14.6f%-14.6f%-16.6f%-16.6f%-16.6f\n' \
                                    %(k*each, max_ene, min_ene, avg_ene, max_frc, min_frc, avg_frc))

        atoms_type_dist = calc_dist(atoms, coord[k], cell[k][0], cell[k][1], cell[k][2])
        dist = []
        atom_type_pair_tot = []
        for key in atoms_type_dist:
          dist.append(min(atoms_type_dist[key]))
          atom_type_pair_tot.append(key)
        min_dist = min(dist)
        min_dist_index = dist.index(min_dist)
        atom_type_pair = atom_type_pair_tot[min_dist_index]

        if ( max_frc <= success_force_conv ):
          success_frames_ij = success_frames_ij + 1
        else:
          if ( abs(max_frc-success_force_conv) <= 0.05 ):
            success_devi_frames_ij = success_devi_frames_ij + 1
          atom_cov_radii_plus = atom.get_atom_cov_radius(atom_type_pair[0]) + \
                                atom.get_atom_cov_radius(atom_type_pair[1])
          if ( min_dist > atom_cov_radii_plus*0.7 and max_frc < max_force_conv ):
            choosed_index.append(k)

      linecache.clearcache()
      success_frames_i.append(success_frames_ij)
      success_devi_frames_i.append(success_devi_frames_ij)
      struct_index_i[j] = choosed_index
      model_devi_file.close()

    tot_frames.append(tot_frames_i)
    success_frames.append(success_frames_i)
    success_devi_frames.append(success_devi_frames_i)
    struct_index[i] = struct_index_i

  success_ratio_sys = []
  tot_frames_sys = []
  success_frames_sys = []
  success_devi_frames_sys = []
  for i in range(len(tot_frames)):
    success_ratio_sys.append(float(float(sum(success_frames[i]))/float(sum(tot_frames[i]))))
    tot_frames_sys.append(sum(tot_frames[i]))
    success_frames_sys.append(sum(success_frames[i]))
    success_devi_frames_sys.append(sum(success_devi_frames[i]))

  success_ratio = float(float(sum(success_frames_sys))/float(sum(tot_frames_sys)))
  success_devi_ratio = float(float(sum(success_devi_frames_sys))/float(sum(tot_frames_sys)))

  return struct_index, success_ratio_sys, success_ratio, success_devi_ratio

def calc_dist(atoms, coord, a_vec, b_vec, c_vec):

  '''
  calc_dist: calculate distance between different atom types
  Args:
    atoms: 1-d string list, dim = numb of atoms
      Example: ['O','H','H']
    coord: 2-d float list, dim = (numb of atoms)*3
      Example: [[3.86,-4.75,5.51], [4.21,-5.56,6.00], [3.09,-4.32,5.91]]
    a_vec, b_vec, c_vec: 1-d array, dim = 3
      They are triclinic cell vector.
      Example: array([Lx,0.0,0.0]), array([xy,Ly,0.0]), array([xz,yz,Lz])
  Returns:
    atoms_type_dist: dictionary, dim = numb of atom-type pairs
      Example : {(O,H):[1.0,1.0], (H,H):[1.5]}
  '''

  atoms_type = data_op.list_replicate(atoms)
  atoms_type_coord = []

  for i in range(len(atoms_type)):
    atom_type_i_coord = []
    for j in range(len(atoms)):
      if ( atoms[j] == atoms_type[i] ):
        atom_type_i_coord.append(coord[j])
    atoms_type_coord.append(atom_type_i_coord)

  atoms_type_dist = OrderedDict()

  for i in range(len(atoms_type_coord)):
    for j in range(len(atoms_type_coord)):
      if ( i <= j ):
        atom_type_pair = (atoms_type[i], atoms_type[j])
        atom_type_i_coord = atoms_type_coord[i]
        atom_type_j_coord = atoms_type_coord[j]
        coord_i = []
        coord_j = []
        for k in range(len(atom_type_i_coord)):
          for l in range(len(atom_type_j_coord)):
            if ( k <= l and atom_type_i_coord[k] != atom_type_j_coord[l] ):
              coord_i.append(atom_type_i_coord[k])
              coord_j.append(atom_type_j_coord[l])
        if ( coord_i != [] and coord_j != [] ):
          dist = geometry.calculate_distance(np.asfortranarray(coord_i, dtype='float32'), \
                                             np.asfortranarray(coord_j, dtype='float32'), \
                                             np.asfortranarray(a_vec, dtype='float32'), \
                                             np.asfortranarray(b_vec, dtype='float32'), \
                                             np.asfortranarray(c_vec, dtype='float32'))
          atoms_type_dist[atom_type_pair] = dist

  return atoms_type_dist

def calc_force_devi(force_a, force_b):

  '''
  calc_force_devi: calculate force deviation
  Args:
    force_a: 2-d list, dim = N*3
    force_b: 2-d list, dim = N*3
  Returns:
    deviation: 1-d list, dim = N
  '''

  deviation = []
  for i in range(len(force_a)):
    vec_1 = np.array(force_a[i])
    vec_2 = np.array(force_b[i])
    deviation.append(np.sqrt(np.sum(np.square(vec_1 - vec_2))))

  #force_a_norm = np.linalg.norm(force_a, axis=1)
  #force_b_norm = np.linalg.norm(force_b, axis=1)
  #deviation = np.abs(force_a_norm - force_b_norm)

  return deviation

if __name__ == '__main__':
  from DPFlow.deepff import model_devi
  from DPFlow.tools import get_cell

  #Test choose_lmp_str function
  work_dir = '/home/lujunbo/WORK/Deepmd/DPFlow/co2/md_mtd'
  iter_id = 17
  atoms_type_multi_sys = {0: {'C': 1, 'O': 2}, 1: {'C': 1, 'O': 2}}
  atoms_num_tot = {0:3,1:3}
  force_conv = 0.05
  struct_index = choose_lmp_str(work_dir, iter_id, atoms_type_multi_sys, atoms_num_tot, force_conv)
  print (struct_index)

  exit()
  #Test calc_dist function
  a, b, c = get_cell.get_triclinic_cell_six([10.0,10.0,10.0,0.0,0.0,0.0])
  atoms = ['C','O','O']
  coord = [[9.99974000,0.00093734,9.99935000], [9.99750000,0.00049800,8.83826000], [0.00269797,9.99880000,1.16223000]]
  atoms_type_dist = model_devi.calc_dist(atoms, coord, a, b, c)
  print (atoms_type_dist)
