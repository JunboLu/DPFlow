from matplotlib import pyplot as plt

import csv
import linecache
import numpy as np

#The stride of sum_hill is 400, so the time is 400*100*0.5=20 ps for each sum_hill;
file_name = ['fes_20.dat', 'fes_30.dat', 'fes_40.dat', 'fes_50.dat']
label = ['400ps', '600ps', '800ps', '1000ps']

## Free energy is in kj/mol

def read_file(file_name):

  x = []
  y = []

  for i in range(300):
    line = linecache.getline(file_name, 6+i)
    line_split = line.split(' ')
    a = []
    for j in range(len(line_split)):
      if ( line_split[j] != '' ):
        a.append(line_split[j])
    x.append(float(a[0]))
    y.append(float(a[1]))

  return x, y

x = []
y = []
for i in file_name:
  x_i, y_i = read_file(i)
  x.append(x_i)
  y.append(y_i)

for i in range(len(x)):
  fes_file = 'fes_' + str(i) + '.csv'
  with open(fes_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for j in range(len(x[i])):
      writer.writerow([x[i][j], y[i][j]])

plt.figure(figsize=(12,8))
for i in range(len(x)):
  plt.plot(x[i], y[i], label=label[i])
plt.legend()
plt.show()


