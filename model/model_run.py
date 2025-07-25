#! /usr/env/bin python

import sys
from DPFlow.tools import log_info
from DPFlow.tools import read_input
from DPFlow.tools import data_op
from DPFlow.model import droplet

work_dir = str(sys.argv[1])
inp_file = str(sys.argv[2])
model_type = str(sys.argv[3])

log_info.log_logo()

print (data_op.str_wrap('MODEL| PROGRAM STARTED IN %s' %(work_dir), 80), flush=True)
print ('MODEL| Input file name %s\n' %(inp_file), flush=True)

model_type_param = read_input.dump_info(work_dir, inp_file, [model_type])

if ( model_type == 'droplet' ):
  droplet.droplet_run(model_type_param[0], work_dir)

