#! /env/bin/python

import os
import sys
import linecache
from collections import OrderedDict
from DPFlow.deepff import write_data
from DPFlow.tools import file_tools
from DPFlow.tools import data_op

input_file = input("Please enter the DPFlow input file name: ")
iter_num = input("Please enter the maximum iteration number: ")
iter_num = int(iter_num)

work_dir = os.getcwd()

line_num = file_tools.grep_line_num('type_map', input_file, work_dir)
line = linecache.getline(input_file, line_num[0])
line_split = data_op.split_str(line, ' ', '\n')
tot_atoms_type = []

for i in range(len(line_split)-1):
  tot_atoms_type.append(line_split[i+1])

tot_atoms_type_dic = OrderedDict()
for i in range(len(tot_atoms_type)):
  tot_atoms_type_dic[tot_atoms_type[i]] = i

write_data.write_active_data(work_dir, iter_num, tot_atoms_type_dic)
