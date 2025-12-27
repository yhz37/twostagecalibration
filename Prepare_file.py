# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:23:57 2024

@author: hp
"""
import xml.etree.ElementTree as ET
import shutil
import os
import numpy as np
import re
import time
from multiprocessing import current_process
import glob

def copy_and_modify_files(base_path, matrix, elastanceMatrix,LA_pressure,simulations_folder_path = None,number='compl'):
    
    if number == 'compl':
        # Get current process name
        process_id = current_process().name
        # Get the current timestamp
        timestamp = int(time.time() * 1000)  # Milliseconds for higher resolution
        
        if simulations_folder_path == None:
            simulations_folder_path = os.path.dirname(base_path)
        new_directory = os.path.join(simulations_folder_path, f'scalar_{process_id}_{timestamp}')
    elif number == 'sim':
        
        if simulations_folder_path == None:
            simulations_folder_path = os.path.dirname(base_path)
            
        i=1
        new_directory = os.path.join(simulations_folder_path, f'scalar_{i}')
        while os.path.exists(new_directory) or glob.glob(os.path.join(simulations_folder_path, f'scalar_{i}_*')):
            new_directory = os.path.join(simulations_folder_path, f'scalar_{i+1}')
            i += 1

    os.makedirs(new_directory)

    # copy file from base path to new directory
    shutil.copytree(base_path, new_directory, dirs_exist_ok=True)

    #faceInfo.dat
    faceInfo_file_path=os.path.join(new_directory, 'faceInfo.dat')
    
    # modify XML
    xml_file_path = os.path.join(new_directory, 'netlist_surfaces.xml')

    modify_parameters(xml_file_path,faceInfo_file_path, matrix,LA_pressure)
    

    # modify elastanceConrol
    elastance_file_path = os.path.join(
        new_directory, 'elastanceControl.py')

    modify_elastance(elastance_file_path, elastanceMatrix)

    # return simulation_path for next step run_simulation
    simulation_path = new_directory
    return simulation_path


def modify_parameters(xml_file_path,faceInfo_file_path, matrix,LA_pressure):
    
    with open(faceInfo_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    face_dic = {}
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) > 1:
            key = parts[-1]  # read the last part of each line
            face_dic[key] = index
    
    
    # original order to read order
    original_order = {'outflow_Aorta':0,'inflow_Aorta':1,'outflow_LM_LAD':2,'outflow_OM1':3,'outflow_OM2':4,'outflow_LCx':5,'outflow_AM':6,'outflow_RCA':7}

    read_order = {key: value for key, value in face_dic.items() if key != 'outflow_L_cathter'}


    sorted_order = {k: v for k, v in sorted(read_order.items(), key=lambda item: item[1])}

    sorted_values = [original_order[key] for key in sorted_order]

    matrix = matrix[sorted_values]


   
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for circuit in root.findall(".//circuit"):
        circuit_index = int(circuit.find("circuitIndex").text) - 1
        components = circuit.findall(".//component")

        for component in components:
            component_index = int(component.find("index").text) - 1
            parameter_value = matrix[circuit_index, component_index]

            component.find(".//parameterValue").text = str(parameter_value)

    tree.write(xml_file_path)
    # print("\nXML file has been updated")
    # print(matrix)
    
    
    sim_folder_dir = os.path.dirname(xml_file_path)
    faceInfo_file_path=os.path.join(sim_folder_dir, 'faceInfo.dat')
    
    with open(faceInfo_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    heart_circuit_idx = face_dic['inflow_Aorta']
    
    circuits = root.findall(".//circuit")
    for circuit in circuits:
        circuit_index = int(circuit.find("circuitIndex").text)
        if circuit_index == heart_circuit_idx+1:
            nodes = circuit.findall(".//node")
            
            for node in nodes:
                index = int(node.find("index").text)
                if index == 5:
                    node.find("initialPressure").text = str(LA_pressure)
                    # print(f"LA pressure has been updated to {LA_pressure}")
                    break


    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
    # print(f"LA pressure has been updated to {LA_pressure}")




def modify_elastance(elastance_file_path, elastanceMatrix):
    # Check if elastanceMatrix exists and is not empty
    if elastanceMatrix:
        # Read the original elastance_control file
        with open(elastance_file_path, 'r') as file:
            script_content = file.read()

        # Define the parameters to modify and their new values
        parameters_to_modify = {
            "m_timeToMaximumElastance": elastanceMatrix[0],
            "m_timeToRelax": elastanceMatrix[1],
            "m_minimumElastance": elastanceMatrix[2],
            "m_maximumElastance": elastanceMatrix[3],
        }

        # Iterate through the parameters and replace them in the script
        for parameter, value in parameters_to_modify.items():
            pattern = rf"{parameter}\s*=\s*[\d.e+-]+"
            replacement = f"{parameter} = {value}"
            script_content = re.sub(pattern, replacement, script_content)

        # Write the new script
        with open(elastance_file_path, 'w') as file:
            file.write(script_content)

        # print("\nelastanceControl file has been updated")
        # print(elastanceMatrix)
    else:
        print("elastanceMatrix is empty or does not exist")
       

if __name__ == "__main__":

    base_path = r"C:\Users\hp\Desktop\optimaze\base_heart\new_baseheart"

    matrix = np.array([[0.005206897188488418, 0.15918831092429198 ,12.997010223417973, None, None],  # Aorta
        [1e-05, 1e-05, 1e-05, 0.000273, 3e-06],  # Heart[L_AV,D_AV,L_MV,D_MV,T_LV resistance]
        [1, 1.86, 17.325, 0.00072, 0.021],  # LM/LAD
        [2, 1.86, 17.325, 0.00144, 0.04],  # OM1
        [3, 1.86, 17.325,  0.00072, 0.021],  # OM2
        [4, 1.86, 17.325, 0.00072, 0.021],  # LCx
        [5, 1.86, 17.325, 0.00072, 0.021],  # AM
        [6, 1.86, 17.325, 0.00102, 0.029]])  # RCA
    
    elastanceMatrix = [10, 2, 3, 4, 1]
    
    LA_pressure=600
    
    copy_and_modify_files(base_path, matrix, elastanceMatrix,LA_pressure)