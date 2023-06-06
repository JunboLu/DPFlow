#! /env/bin/python

import sys
import linecache
import numpy as np
from matplotlib import pyplot as plt

def split_str(str_tmp, space_char, strip_char=''):

  str_tmp_split = str_tmp.split(space_char)
  list_tmp = []
  for i in range(len(str_tmp_split)):
    if ( str_tmp_split[i] != ''):
      list_tmp.append(str_tmp_split[i])

  if ( strip_char != '' ):
    if ( list_tmp[len(list_tmp)-1] == strip_char ):
      list_tmp.remove(list_tmp[len(list_tmp)-1])
    else:
      list_tmp[len(list_tmp)-1] = list_tmp[len(list_tmp)-1].strip(strip_char)

  return list_tmp

colvar_id = input("Please enter the row number of colvar: ")

colvar_id = int(colvar_id)

whole_line_num = len(open('COLVAR', 'r').readlines())

time = []
colvar = []

for i in range(whole_line_num-1):
  line = linecache.getline('COLVAR', i+2)
  line_split = split_str(line, ' ', '\n')
  time.append(float(line_split[0]))
  colvar.append(float(line_split[colvar_id-1]))

plt.figure(figsize=(12,8))
plt.plot(np.array(time), np.array(colvar))
plt.show()
plt.savefig('./colvar_evo.png')
plt.close()

plt.hist(np.array(colvar))
plt.show()
plt.savefig('./colvar_hist.png')
plt.close()
