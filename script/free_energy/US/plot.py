from matplotlib import pyplot as plt

import csv
import linecache
import numpy as np

###################################################
#Please change the input parameter
align_ref_point = 1
file_name = 'fes_phi_catr.dat'
###################################################

bin_num = len(open(file_name, 'r').readlines()) - 6

x = []
fes = []

line = linecache.getline(file_name, 6+align_ref_point)
line_split = line.split(' ')
a = []
for i in range(len(line_split)):
  if ( line_split[i] != '' ):
    a.append(line_split[i])

fes_ref = float(a[1])

for i in range(bin_num):
  line = linecache.getline(file_name, 7+i)
  line_split = line.split(' ')
  a = []
  for j in range(len(line_split)):
    if ( line_split[j] != '' ):
      a.append(line_split[j])
  x.append(float(a[0]))
  fes.append(float(a[1])-fes_ref)

#fes is in kj/mol
fes_file = 'fes.csv'
with open(fes_file, 'w') as csvfile:
  writer = csv.writer(csvfile)
  for i in range(len(x)):
    writer.writerow([x[i], fes[i]])

plt.figure(figsize=(12,8))
plt.plot(x, fes)
plt.show()


