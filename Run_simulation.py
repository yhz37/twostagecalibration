# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:31:27 2024

@author: hp
"""


import subprocess
import os
import glob
from sys import platform



def Run_simulation(simulation_path):
    # command1 = 'cd "C:/Program Files/CRIMSON 2024.03.03/bin/Python/lib/site-packages/CRIMSONSolver/SolverStudies/flowsolver"'
    # command2 = f'mysolver "{simulation_path}"'

    # combined_command = f'{command1} && {command2}'

    # Open a file to redirect the output
    while not glob.glob(os.path.join(simulation_path, '*-procs-case')):  # Infinite loop to keep trying until the command succeeds
        with open(os.path.join(simulation_path, 'simulation_output.log'), 'w') as output_file:
            try:
                # Redirect standard output and standard error to the file
                if platform == "win32":
                    command2 = f'mysolver "{simulation_path}"'
                    subprocess.run(command2, shell=True, check=True, text=True, input='1',cwd="C:/Program Files/CRIMSON 2024.03.03/bin/Python/lib/site-packages/CRIMSONSolver/SolverStudies/flowsolver", stdout=output_file,
                                   stderr=subprocess.STDOUT)
                elif platform == "linux": 
                    subprocess.run(f'cd {simulation_path}', shell=True)
                    subprocess.run('mpirun -np 1 /scratch/figueroc_root/figueroc0/shared_data/sw/scalar_cfs/bin/flowsolver ./solver.inp',
                           shell=True, check=True, stdout=output_file, stderr=subprocess.STDOUT, cwd=simulation_path)

            except subprocess.CalledProcessError as e:
                print(f"Simulation failed: {e}")

    result_path = glob.glob(os.path.join(simulation_path, '*-procs-case'))[0]   # *-procs-case each procs can use
    
    return result_path



if __name__ == "__main__":
    
    simulation_path = r"D:\Crimson_Sim\0D\Baseline\scalar_319"
    
    Run_simulation(simulation_path)

    

    