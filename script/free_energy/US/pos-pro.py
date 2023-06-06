import math
import plumed
import linecache
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import concurrent.futures

def wham(bias,
        *,
        frame_weight=None,
        traj_weight=None,
        T: float = 1.0,
        maxiter: int = 1000,
        threshold: float = 1e-40,
        verbose: bool = False):

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    # default values
    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    # divide by T once for all
    shifted_bias = bias/T
    # track shifts
    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis,:]
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:,np.newaxis]

    # do exponentials only once
    expv = np.exp(-shifted_bias)

    Z = np.ones(ntraj)

    Zold = Z

    if verbose:
        sys.stderr.write("WHAM: start\n")
    for nit in range(maxiter):
        # find unnormalized weights
        weight = 1.0/np.matmul(expv, traj_weight/Z)*frame_weight
        # update partition functions
        Z = np.matmul(weight, expv)
        # normalize the partition functions
        Z /= np.sum(Z*traj_weight)
        # monitor change in partition functions
        eps = np.sum(np.log(Z/Zold)**2)
        Zold = Z
        if verbose:
            sys.stderr.write("WHAM: iteration "+str(nit)+" eps "+str(eps)+"\n")
        if eps < threshold:
            break
    nfev=nit
    logW = np.log(weight) + shifts1

    if verbose:
        sys.stderr.write("WHAM: end")

    return {"logW":logW, "logZ":np.log(Z)-shifts0, "nit":nit, "eps":eps}


#start
limit=(input("input the lower and upper bound"))
limit=np.asarray(limit.split(),dtype='float64')
interval=int((input("how many window you wanna option")))
############################################################
#The unit of kb*T is kj/mol
kb=0.008314
############################################################
#T is temperature, need to change by user
T=50
############################################################
#sum up the individual xyz into a sum xyz

for i in range(interval+1):
  line = linecache.getline('./%d/dump.%d.xyz' %(i,i), 1)
  atoms_num = int(line.strip('\n'))
  frames_num = math.ceil(len(open('./%d/dump.%d.xyz' %(i,i)).readlines())/atoms_num)
  cmd = "sed -ie '1,%dd' dump.%d.xyz" %(int(frames_num/2)*(atoms_num+2), i)
  subprocess.run(cmd, cwd='./%d' %(i), shell=True)

subprocess.run("bash -c 'for i in {0..%d}; do cat ./$i/dump.$i.xyz >>./dump-sum.xyz;done'" % (interval),shell=True)

for i in range(interval+1):
  cmd = "mv dump.%d.xyze dump.%d.xyz" %(i,i)
  subprocess.run(cmd, cwd='./%d' %(i), shell=True)

#create the non-dump.xyz plumed file again for post-cal.
############################################################
#need to modify to your system, change the plumed.dat
increment=float((limit[1]-limit[0])/interval)
at=[]
for i in range(interval+1):
  at.append(limit[0]+increment*i)
at=np.array(at)
for i in range(interval+1):
    with open("plumed_" + str(i) + ".dat","w") as f:
        print("""
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs
MOLINFO STRUCTURE=./c2h6.pdb
#RESTART
phi: TORSION ATOMS=5,1,2,8

bb: RESTRAINT ARG=phi KAPPA=200.0 AT={}

PRINT STRIDE=100 FILE=colvar_multi_{}.dat ARG=phi,bb.bias
""".format(at[i],i,i),file=f)
############################################################
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for i in range(interval+1):
        executor.submit(subprocess.run,"plumed driver --plumed plumed_{}.dat --ixyz dump-sum.xyz --trajectory-stride 100".format(i),shell=True)

col=[]
for i in range(interval+1):
    col.append(plumed.read_as_pandas("colvar_multi_" + str(i)+".dat"))
# notice that this is the concatenation of 32 trajectories with 2001 frames each
#后期可能涉及到改每个cv在总col的占比，只影响画图，不影响最终结果
    #plt.plot(col[i].time[2001*i:2001*(i+1)],col[i].phi[2001*i:2001*(i+1)],"x")
#plt.xlabel("$\phi$")
#plt.ylabel("$\psi$")
#plt.show()
#plt.savefig("cv-time.png")
# in this graph you can see which region was sampled by each simulation
#plt.close()
bias=np.zeros((len(col[0]["bb.bias"]),interval+1))
for i in range(interval+1):
    bias[:,i]=col[i]["bb.bias"][-len(bias):]
w=wham(bias,T=kb*T)
plt.plot(w["logW"])
plt.show()
plt.savefig("step-logW.png")
plt.close()
colvar=col[0]
colvar["logweights"]=w["logW"]
plumed.write_pandas(colvar,"bias_multi.dat")

with open("plumed_multi.dat","w") as f:
    print("""
# vim:ft=plumed
phi: READ FILE=bias_multi.dat VALUES=phi IGNORE_TIME
lw: READ FILE=bias_multi.dat VALUES=logweights IGNORE_TIME

# use the command below to compute the histogram of phi
# we use a smooth kernel to produce a nicer graph here
hhphi: HISTOGRAM ARG=phi GRID_MIN=-pi GRID_MAX=pi GRID_BIN=600 BANDWIDTH=0.05
ffphi: CONVERT_TO_FES GRID=hhphi TEMP=%d #
DUMPGRID GRID=ffphi FILE=fes_phi_cat.dat

# we use a smooth kernel to produce a nicer graph here
hhphir: HISTOGRAM ARG=phi GRID_MIN=-pi GRID_MAX=pi GRID_BIN=600 BANDWIDTH=0.05 LOGWEIGHTS=lw
ffphir: CONVERT_TO_FES GRID=hhphir TEMP=%d #
DUMPGRID GRID=ffphir FILE=fes_phi_catr.dat


""" %(T,T),file=f)

subprocess.run("plumed driver --noatoms --plumed plumed_multi.dat --kt {}".format(kb*T),shell=True)

colvar=plumed.read_as_pandas("bias_multi.dat")
plt.plot(colvar.time,colvar.phi,"x",label="phi")
plt.xlabel("time")
plt.ylabel("$\phi$")
plt.legend()
plt.show()
plt.savefig("final-time-cv.png")
plt.close()

fes_phib=plumed.read_as_pandas("fes_phi_cat.dat").replace([np.inf, -np.inf], np.nan).dropna()
plt.plot(fes_phib.phi,fes_phib.ffphi,label="biased")
fes_phir=plumed.read_as_pandas("fes_phi_catr.dat").replace([np.inf, -np.inf], np.nan).dropna()
plt.plot(fes_phir.phi,fes_phir.ffphir,label="reweighted")
plt.legend()
plt.xlim((limit[0],limit[1]))
plt.xlabel("$\phi$")
plt.ylabel("$F(\phi)$")
plt.show()
plt.savefig("final-cv-fes.png")
plt.close()
