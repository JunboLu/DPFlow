<div align="left">
  <img src="https://github.com/JunboLu/DPFlow/blob/main/doc/logo.png" height="160px"/>
</div>

# DPFlow - A workflow to generate deep potential force field
---------

DPFlow is a code to:  
(1) Optimize neural network force field  
(2) Analyze CP2K and LAMMPS trajectory  

The code is mainly written in Python and Fortran.  

Email: lujunbo15@gmail.com
  
# Installation for DPFlow

* Prerequisites
   - Python 3.5 or higher
   - Numpy 1.8.0 or higher

   Suggestion: as we will use deepmd-kit software, we recommend users could  
   install deepmd-kit at firsti: https://github.com/deepmodeling/deepmd-kit.   
   Then add the environmental variable of deepmd-kit, it will include python   
   and numpy.  

* Install GNU parallel

    Download GNU parallel source code from https://www.gnu.org/software/parallel/  
    tar -jxvf parallel-latest.tar.bz2  
    cd parallel-latest  
    ./configure --prefix=parallel_install_path  
    make  
    make install  

* Compile core module
  
    git clone https://github.com/JunboLu/DPFlow.git  
    !Caution: When downloading DPFlow through zip version, please change the  
    name "DPFlow-main" to "DPFlow" after you unzip it.  
    cd DPFlow_directory/lib  
    change directory of f2py in Makefile  
    !Caution: If your gcc version is low, f2py cannot compile core code please  
    update your gcc up to 6.3.  
    make  

* Environmental variable

    !Caution: You may need to add environments of gcc
    export PYTHONPATH=../DPFlow_directory:$PYTHONPATH  
    change python_exe and DPFlow directory in DPFlow_directory/bin/DPFlow file  
    export PATH=DPFlow_directory/bin:$PATH  

# How to use 
* DPFlow is an user-friendly code.  

  The input files are in example directory.  
  Users just need to run:  
  DPFlow input.inp  
