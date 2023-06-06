from matplotlib import pyplot as plt

import csv
import linecache
import numpy as np

temp = input("Please enter the temperature: ")
hist_file = input("Please enter the histogram file name: ")

whole_line_num = len(open(hist_file, 'r').readlines())
bins = whole_line_num-6
temp = float(temp)

#The unit of kbt is kj/mol
kbt = temp*1.38064852*6.022140857/1000.0
x = []
prob = []
fes = []

for i in range(bins):
  line = linecache.getline(hist_file, 7+i)
  line_split = line.split(' ')
  a = []
  for j in range(len(line_split)):
    if ( line_split[j] != '' ):
      a.append(line_split[j])
  x.append(float(a[0]))
  prob.append(float(a[1]))

#In numpy, log() is ln, and log10() is log in math.
for i in range(len(prob)):
  fes.append(-kbt*np.log(prob[i]))

#We will align the fes
fes_align = []
min_fes = min(fes)
for i in range(len(fes)):
  fes_align.append(fes[i]-min_fes)
#fes is in kj/mol
fes_file = 'fes.csv'
with open(fes_file, 'w') as csvfile:
  writer = csv.writer(csvfile)
  for i in range(len(x)):
    writer.writerow([x[i], fes_align[i]])

plt.figure(figsize=(12,8))
plt.plot(x, fes_align)
plt.show()
