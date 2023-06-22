#!/usr/bin/env python
'''
srun_multi.py
  script to srun a PYMC job
Author:  M.E. Glinsky, Sandia National Laboratories
Date: 24 February 2018
Arguments:
  file_list -- file with list of parameters to execute (default is "~/files.txt")
               one list of arguments per line in the file
  multi_script -- script to execute multiple times (default is "run_snlxri.py")
Shell eviormental variables used:
  PPN -- processors per node (default 16)
  NODES_DEBUG -- number of nodes (default 1)
  HYDRA_SCN -- flag for it being the SCN (default False)
  IS_MAC -- flag for it being a Mac (default False)
'''

import sys, os

def main():
    # process the arguments
    #print(len(sys.argv),sys.argv)  # for debug
    PPN = int(os.getenv('PPN', '16'))
    NODES_DEBUG = int(os.getenv('NODES_DEBUG', '1'))
    HYDRA_SCN = os.getenv('HYDRA_SCN', 'False')
    IS_MAC = os.getenv('IS_MAC', 'False')
    if len(sys.argv) == 1:
        file_list = os.getenv('ARG1', '~/tmp/files.txt')
        multi_script = os.getenv('ARG2', 'pySAMM_run.py')
    elif len(sys.argv) == 2:
        file_list = sys.argv[1]
        multi_script = os.getenv('ARG2', 'pySAMM_run.py')
    elif len(sys.argv) == 3:
        file_list = sys.argv[1] 
        multi_script = sys.argv[2]
    else:
        print('ERROR: srun_multi with up to two parameters -- file_list, multi_script')
        return -1

    if HYDRA_SCN == 'False' and IS_MAC == 'False':
        cmd = 'srun -N ' + str(NODES_DEBUG) + ' -n' + str(PPN) + ' multi ' + multi_script  + ' -multi_file ' + file_list
    else:
        cmd = 'mpiexec -np ' + str(NODES_DEBUG*PPN) + ' multi ' + multi_script + ' -multi_file ' + file_list
    print cmd
    os.system(cmd)
        
#----------------------------------------------
if __name__ == "__main__":
    main()
