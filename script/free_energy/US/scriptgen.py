import numpy as np
import subprocess
limit=(input("input the lower and upper bound"))
limit=np.asarray(limit.split(),dtype='float64')
interval=int((input("how many window you wanna option")))
increment=float((limit[1]-limit[0])/interval)
at=[]
for i in range(interval+1):
  at.append(limit[0]+increment*i)
at=np.array(at)
print(at)
for i in range(interval+1):
    with open("plumed_" + str(i) + ".dat","w") as f:
        print("""
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs
MOLINFO STRUCTURE=../c2h6.pdb
#RESTART
phi: TORSION ATOMS=5,1,2,8

bb: RESTRAINT ARG=phi KAPPA=200.0 AT={}

PRINT STRIDE=100 FILE=colvar_multi_{}.dat ARG=phi,bb.bias
DUMPATOMS FILE=dump.{}.xyz ATOMS=@mdatoms STRIDE=100
""".format(at[i],i,i),file=f)

subprocess.run("bash -c 'for i in {0..%d}; do mkdir $i ;mv plumed_$i.dat ./$i/plumed.dat ;done'" % (interval),shell=True)


