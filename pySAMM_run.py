#!/usr/bin/env python

'''
Python script to execute snlxri using MPI. Command should be:
     mpiexec -np str(NODES*PPN) multi run_snlxri.py -multi_break 3 run_name_0 parm_00 parm_01 run_name_1 parm_10 parm_11  (for SCN)
     mpirun -N str(NODES*PPN) multi run_snlxri.py run_name_0 run_name_1 run_name_2 run_name_3 
     srun -N str(NODES) -n str(PPN) -p pdebug multi run_snlxri.py  -multi_file parm_file.txt   (for RZ) 
     srun -N 1 -n 36 -p pdebug multi run_snlxri.py  -multi_file ~/snlxri_runs/lists/run_names_v2.txt   (for RZ run multiple runs) 
Can also run nohup:
    nohup run_snlxri.py run_name gas_type scalar dope_gas vol_frac_dope  > stdout.out & (scalar run)
    nohup run_snlxri.py run_name gas_type srun-debug dope_gas vol_frac_dope  > stdout.out &  (run with MPI in debug queue)
    plot_snlxri.py run_name_0
Can run with sbatch:
    snlxri_sbatch run_name gas_type dope_gas vol_frac_dope
'''

# print 'here before sys'
import sys, os
# print 'here before pygains.multi'
from pySAMM_v2 import *

run_type = 'srun'

if len(sys.argv) == 2:
    fname = sys.argv[1]
else:
    print('ERROR: run_pySAMM with fname only')

print 'Running pySAMM input for deck: ' + fname

# set the display so that it can run headless
if run_type == 'srun':
    DISPLAY = os.getenv('DISPLAY', ':99')
    print 'DISPLAY = ' + DISPLAY + ', setting DISPLAY to Xvfb headless :99'
    cmd = 'Xvfb :99 &'
    print cmd
    os.system(cmd)
    os.putenv('DISPLAY',':99')
    print 'temporarily set DISPLAY ='
    os.system('echo $DISPLAY')
    sys.stdout.flush()

run_pySAMM(fname=fname, save_outputs=1)

if run_type == 'srun':
    print 'restoring DISPLAY ='
    os.putenv('DISPLAY',DISPLAY)
    os.system('echo $DISPLAY')
    sys.stdout.flush()

print 'exiting run_pySAMM.py'