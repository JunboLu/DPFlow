#! /usr/env/bin python
import os
import sys
import linecache
import subprocess
import numpy as np
from DPFlow.tools import call
from DPFlow.tools import log_info
from DPFlow.tools import traj_info
from DPFlow.tools import data_op
from DPFlow.tools import file_tools
from DPFlow.tools import revise_cp2k_inp

##############################################
print('\nNotes: The files of COLVAR, thcl4.inp (template of CP2K input), thcl4.xyz (coordinate trajectory file) and thcl4.cell (cell file) must exist in the current folder before running this code!\n')
input_file = input("Please enter the xyz trajectory file name, eg. thcl4 : ")
choose_num = input("Please enter the number of choosed frames, eg. 3000 : ")
use_prev_wfn = input("Please enter whether use wavefunction of previous step (true or false): ")
CV_min = input("Please enter the minimum of CV, eg. 8.0 : ")
CV_max = input("Please enter the maximum of CV, eg. 10.0 : ")
CV_bin = input("Please enter the number of bins for CV, eg. 100 : ")
stride = input("Please enter the stride of saving the xyz trajectory file, eg. 1000 : ")
##############################################

choose_num = int(choose_num)
use_prev_wfn = data_op.str_to_bool(use_prev_wfn)
CV_min = float(CV_min)
CV_max = float(CV_max)
CV_bin = int(CV_bin)
stride = int(stride)

work_dir = os.getcwd()
subprocess.run('rm -rf data', cwd=work_dir, shell=True)
subprocess.run('mkdir data', cwd=work_dir, shell=True)
data_dir = ''.join((work_dir, '/data'))

input_tmp = ''.join((input_file, '_tmp'))
#upper_file_name_abs = file_tools.upper_file(input_file, work_dir)
upper_file_name_abs = file_tools.upper_file('%s.inp'%input_file, work_dir)
space_file_name_abs = file_tools.space_file(upper_file_name_abs, ' ', work_dir)

#proj_name = revise_cp2k_inp.get_proj_name(input_file, work_dir)
proj_name = input_file

#traj_coord_file = ''.join((proj_name, '-pos-1.xyz'))
traj_coord_file = ''.join((proj_name, '.xyz'))
traj_box_file = ''.join((proj_name, '.cell'))
traj_COLVAR_file = 'COLVAR'
cp2k_input_template = ''.join((proj_name, '.inp'))

if ( not os.path.exists(traj_coord_file) ):
  log_info.log_error('Input error: Coordinate trajectory file %s does not exist!' %(traj_coord_file))
  exit()
if ( not os.path.exists(traj_box_file) ):
  log_info.log_error('Input error: Box file %s does not exist!' %(traj_box_file))
  exit()
if ( not os.path.exists(traj_COLVAR_file) ):
  log_info.log_error('Input error: COLVAR file %s does not exist!' %(traj_COLVAR_file))
  exit()
if ( not os.path.exists(cp2k_input_template) ):
  log_info.log_error('Input error: Template file of CP2K input %s does not exist!' %(cp2k_input_template))
  exit()

input_tmp_file = open(input_tmp, 'w')
input_tmp_file.write('&GLOBAL\n')
input_tmp_file.write('  PROJECT %s\n' %(proj_name))
input_tmp_file.write('  RUN_TYPE ENERGY_FORCE\n')
input_tmp_file.write('  PRINT_LEVEL LOW\n')
input_tmp_file.write('&END GLOBAL\n')
input_tmp_file.write('\n')

input_tmp_file.write('&FORCE_EVAL\n')
input_tmp_file.write('  METHOD Quickstep\n')
line_num_1 = file_tools.grep_line_num('&DFT', space_file_name_abs, work_dir)
line_num_2 = file_tools.grep_line_num('&END DFT', space_file_name_abs, work_dir)

#print(space_file_name_abs)
#print(line_num_1, line_num_2)

for i in range(line_num_1[0], line_num_2[0]+1, 1):
#for i in range(line_num_1, line_num_2+1, 1):
  #line = linecache.getline(input_file, i)
  line = linecache.getline('%s.inp'%input_file, i)
  input_tmp_file.write(line)
input_tmp_file.write('\n')

input_tmp_file.write('  &PRINT\n')
input_tmp_file.write('    &FORCES\n')
input_tmp_file.write('      FILENAME\n')
input_tmp_file.write('    &END FORCES\n')
input_tmp_file.write('  &END PRINT\n')
input_tmp_file.write('\n')

input_tmp_file.write(' &SUBSYS\n')
input_tmp_file.write('    &PRINT\n')
input_tmp_file.write('      &ATOMIC_COORDINATES\n')
input_tmp_file.write('        FILENAME\n')
input_tmp_file.write('      &END ATOMIC_COORDINATES\n')
input_tmp_file.write('      &CELL\n')
input_tmp_file.write('        FILENAME\n')
input_tmp_file.write('      &END CELL\n')
input_tmp_file.write('    &END PRINT\n')
input_tmp_file.write('   &CELL\n')
input_tmp_file.write('     @include box\n')
input_tmp_file.write('     PERIODIC XYZ\n')
input_tmp_file.write('   &END CELL\n')
input_tmp_file.write('   &COORD\n')
input_tmp_file.write('      @include coord\n')
input_tmp_file.write('    &END COORD\n')
input_tmp_file.write('\n')

line_num_1 = file_tools.grep_line_num('&KIND', space_file_name_abs, work_dir)
line_num_2 = file_tools.grep_line_num('&END KIND', space_file_name_abs, work_dir)

for i in range(len(line_num_1)): 
  for j in range(line_num_1[i], line_num_2[i]+1, 1):
    #line = linecache.getline(input_file, j)
    line = linecache.getline('%s.inp'%input_file, j)
    input_tmp_file.write(line)
input_tmp_file.write('\n')

'''
for j in range(line_num_1, line_num_2+1, 1):
    #line = linecache.getline(input_file, j)
    line = linecache.getline('%s.xyz'%input_file, j)
    input_tmp_file.write(line)
input_tmp_file.write('\n')
'''
input_tmp_file.write('  &END SUBSYS\n')
input_tmp_file.write('&END FORCE_EVAL\n')

linecache.clearcache()
input_tmp_file.close()

revise_cp2k_inp.revise_basis_file_name(input_tmp, work_dir)
revise_cp2k_inp.revise_pot_file_name(input_tmp, work_dir)
revise_cp2k_inp.revise_dftd3_file_name(input_tmp, work_dir)
revise_cp2k_inp.delete_line('WFN_RESTART_FILE_NAME', input_tmp, work_dir)
revise_cp2k_inp.delete_line('SCF_GUESS', input_tmp, work_dir)
pot_line_num = file_tools.grep_line_num('POTENTIAL_FILE_NAME', input_tmp, work_dir)[0]
scf_line_num = file_tools.grep_line_num('&SCF', input_tmp, work_dir)[0]

atoms_num, pre_base_block, end_base_block, pre_base, frames_num, each, start_frame_id, end_frame_id, time_step = \
traj_info.get_traj_info(traj_coord_file, 'coord_xyz')

total_index = data_op.gen_list(start_frame_id, end_frame_id, each)
total_index_array = np.array(total_index)
if ( choose_num > len(total_index_array) ):
  log_info.log_error('Input error: choose_num is larger than the number of frames in trajectory, please check!')
  exit()


#np.random.shuffle(total_index_array)
#choosed_index = list(total_index_array[0:choose_num])
#choosed_index = sorted(choosed_index)

###################################################################################
#            Calculating the coordination number (CN) for each frame              #
###################################################################################

print("\nStep 1: Loading the COLVAR file to extract the CV of each frame")

COLVAR = np.loadtxt('COLVAR', skiprows = 2, dtype = float, usecols = [0,1])
COLVAR_new = COLVAR[::stride]
np.savetxt("COLVAR_new", COLVAR_new, delimiter=' ',fmt='%s %s')

###################################################################################
#            Selection of structures based on the frequencies                     #
###################################################################################

print("\nStep 2: Selection of structures based on the bins and frequencies")

time = COLVAR_new[:,0]
CN = COLVAR_new[:,1]

#print(CN)
#n, bins, patches = plt.hist(CN, CV_bin)
#freq = n/sum(n)
#print(n/sum(n), n, bins)

count, bins, number = [], [], []

for i in range(CV_bin+1):
  bins.append(CV_min + i*(CV_max - CV_min)/CV_bin)
#print(bins)

for i in range(CV_bin):
  #count.append([time[j] for j in range(len(CN)) if  float(CN[j]) > bins[i] and float(CN[j]) < bins[i+1]])
  #count.append([j for j in range(len(CN)) if  float(CN[j]) > bins[i] and float(CN[j]) < bins[i+1]])
  count.append([])
  num = 0
  for j in range(len(CN)):
    if  float(CN[j]) > bins[i] and float(CN[j]) < bins[i+1]:
      count[i].append(j)
      num += 1
  number.append(num)

#print(count)
#print(number)
#freq = number/sum(number)
freq = [number[i]/sum(number) for i in range(len(number))]
#print(freq)

choosed_index = []
for i in range(CV_bin):
  candidate_num = int(float(freq[i]) * choose_num)
  np.random.shuffle(count[i])
  for j in count[i][0:candidate_num]:
    choosed_index.append(j) 
#print(choosed_index)
#print('The index of selected frames in the trajectory file is: %s'%sorted(choosed_index))
#print(len(choosed_index))

###################################################################################
#            Generation of the CP2K input files for each frame                    #
###################################################################################

print("\nStep 3: Generation of the CP2K input files for the selected frames")

for i in range(len(choosed_index)):
  task_dir = ''.join((data_dir, '/task_', str(i)))
  if ( not os.path.exists(task_dir) ):
    cmd = "mkdir %s" %(''.join(('task_', str(i))))
    call.call_simple_shell(data_dir, cmd)
  inp_file = ''.join((task_dir, '/cp2k.inp'))
  cmd = "cp %s %s" %(input_tmp, inp_file)
  call.call_simple_shell(work_dir, cmd)

  if use_prev_wfn:
    if i != 0:
      cmd = "sed -i '%d s/^/    WFN_RESTART_FILE_NAME ..\/task_%d\/%s-RESTART.wfn\\n/' cp2k.inp" \
             %(pot_line_num+1, i-1, proj_name)
    else:
      cmd = "sed -i '%d s/^/    WFN_RESTART_FILE_NAME .\/%s-RESTART.wfn\\n/' cp2k.inp" \
             %(pot_line_num+1, proj_name)
  else:
    cmd = "sed -i '%d s/^/    WFN_RESTART_FILE_NAME .\/%s-RESTART.wfn\\n/' cp2k.inp" \
           %(pot_line_num+1, proj_name)
  call.call_simple_shell(task_dir, cmd)

  if ( pot_line_num < scf_line_num ):
    cmd = "sed -i '%d s/^/      SCF_GUESS  RESTART\\n/' cp2k.inp" %(scf_line_num+2)
  else:
    cmd = "sed -i '%d s/^/      SCF_GUESS  RESTART\\n/' cp2k.inp" %(scf_line_num+1)
  call.call_simple_shell(task_dir, cmd)

  cood_file_name = ''.join((task_dir, '/coord'))
  box_file_name = ''.join((task_dir, '/box'))
  coord_file = open(cood_file_name, 'w')
  box_file = open(box_file_name, 'w')
  
  for j in range(atoms_num):
    line_ij_num = int((choosed_index[i]-start_frame_id)/each)*(pre_base_block+atoms_num+end_base_block)+pre_base+pre_base_block+j+1
    line = linecache.getline(traj_coord_file, line_ij_num)
    coord_file.write(line)
  coord_file.close()

  #line = linecache.getline(traj_box_file, int((choosed_index[i]-start_frame_id)/each)+2)
  line = linecache.getline(traj_box_file, 1)

  line_split = data_op.split_str(line, ' ')
  box_file.write(''.join(('A  ', line_split[2], '  ', line_split[3], '  ', line_split[4], '\n')))
  box_file.write(''.join(('B  ', line_split[5], '  ', line_split[6], '  ', line_split[7], '\n')))
  box_file.write(''.join(('C  ', line_split[8], '  ', line_split[9], '  ', line_split[10], '\n')))
  box_file.close()

linecache.clearcache()

cmd = 'rm %s' %(space_file_name_abs)
call.call_simple_shell(work_dir, cmd)

cmd = 'rm %s' %(input_tmp)
call.call_simple_shell(work_dir, cmd)

print("\nCongratulation ! Now you can submit these CP2K tasks for calculatig their energies and forces !\n")