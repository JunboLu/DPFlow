#! /env/bin/python

import os
import sys
import linecache
from DPFlow.tools import data_op 
from DPFlow.deepff import process

max_iter = input("Please enter the maximum iteration: ")
force_conv = input("Please enter the force convergence: ")

max_iter = int(max_iter)
force_conv = float(force_conv)

direc = os.getcwd()
lmp_dir = ''.join((direc, '/iter_0/02.lammps_calc'))
sys_num = process.get_sys_num(lmp_dir)
lmp_sys_dir = ''.join((lmp_dir, '/sys_0'))
task_num = process.get_task_num(lmp_sys_dir)

accuracy = []
for i in range(max_iter):
  accurat_num = 0
  total_num = 0
  for j in range(sys_num):
    for k in range(task_num):
      model_devi_file = ''.join((direc, '/iter_', str(i), '/02.lammps_calc/sys_', str(j), '/task_', str(k), '/model_devi.out'))
      whole_line_num = len(open(model_devi_file, 'r').readlines())
      total_num = total_num+whole_line_num-1
      for l in range(whole_line_num-1):
        line = linecache.getline(model_devi_file, l+2)
        line_split = data_op.split_str(line, ' ', '\n')
        if ( float(line_split[4]) < force_conv ):
          accurat_num = accurat_num+1

  accuracy.append(accurat_num/total_num)

for i in range(max_iter):
  print ('The accurate ratio of iter %d is %.2f %%' %(i, accuracy[i]*100))
