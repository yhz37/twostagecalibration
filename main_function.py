# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:20:27 2024

@author: jiyangz

file name:main.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add parent directory to sys.path
sys.path.append(parent_dir)

import re
import numpy as np
from Prepare_file import copy_and_modify_files
from Run_simulation import Run_simulation
import xml.etree.ElementTree as ET
from Postprocess import Postprocess


def extract_matrix_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    circuits = root.findall(".//circuit")

    max_circuit_index = max(int(circuit.find("circuitIndex").text) for circuit in circuits)
    max_component_index = max(int(component.find("index").text) for circuit in circuits for component in circuit.findall(".//component"))

    matrix = np.zeros((max_circuit_index, max_component_index))

    for circuit in circuits:
        circuit_index = int(circuit.find("circuitIndex").text) - 1
        components = circuit.findall(".//component")

        for component in components:
            component_index = int(component.find("index").text) - 1
            parameter_value = float(component.find("parameterValue").text)
            matrix[circuit_index, component_index] = parameter_value
    # print("\nOriginal matrix")
    np.set_printoptions(precision=2)
    # print(matrix)
    
    sim_folder_dir = os.path.dirname(xml_file_path)
    faceInfo_file_path=os.path.join(sim_folder_dir, 'faceInfo.dat')
    
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

    sorted_values = [sorted_order[key] for key in original_order]

    matrix = matrix[sorted_values]
    
    return matrix



def extract_elastanceMatrix(elastance_file_path):
    elastanceMatrix = []

    with open(elastance_file_path, 'r') as file:
        script_content = file.read()

    # Define the parameters to extract
    parameters = [
        "m_timeToMaximumElastance",
        "m_timeToRelax",
        "m_minimumElastance",
        "m_maximumElastance"
    ]

    # Extract the values of the parameters
    for parameter in parameters:
        pattern = rf"{parameter}\s*=\s*([\d.e+-]+)"
        match = re.search(pattern, script_content)
        if match:
            value = float(match.group(1))
            elastanceMatrix.append(value)

    # print("\nOriginal elastanceMatrix")
    # print(elastanceMatrix)
    
    return elastanceMatrix

def extract_LA_pressure(xml_file_path):
    
    sim_folder_dir = os.path.dirname(xml_file_path)
    faceInfo_file_path=os.path.join(sim_folder_dir, 'faceInfo.dat')
    
    with open(faceInfo_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    face_dic = {}
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) > 1:
            key = parts[-1]  # read the last part of each line
            face_dic[key] = index
    heart_circuit_idx = face_dic['inflow_Aorta']
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    circuits = root.findall(".//circuit")
    for circuit in circuits:
        circuit_index = int(circuit.find("circuitIndex").text)
        if circuit_index == heart_circuit_idx+1:
            nodes = circuit.findall(".//node")
            
            for node in nodes:
                node_index = int(node.find("index").text)
                if node_index == 5:
                    LA_pressure = float(node.find("initialPressure").text)
                    # print(f"\nLA pressure: {LA_pressure}")
                    return LA_pressure

# Usage:
# elastanceMatrix = extract_elastanceMatrix('path/to/elastance_control_file')


def main_function(base_path, matrix, elastanceMatrix,LA_pressure,delete=False,heart_plot = None,pvplot = None):

    run = True

    if run:
        simulation_path = copy_and_modify_files(
            base_path, matrix, elastanceMatrix, LA_pressure)
        result_path = Run_simulation(simulation_path)
        result_dic = Postprocess(default_path=result_path,pvplot = pvplot,heart_plot = heart_plot,delete=delete)

    else:
        copy_and_modify_files(base_path, matrix, elastanceMatrix)
        
    return result_dic,result_path


if __name__ == "__main__":

    ### input
    # base_path = r"D:\Crimson_Sim\0D\Baseline\Base_Hyper_DE_9.24_8.8_39_125_75_RpMin_Coronary_Healthy_CorrectPim"
    base_path = r"D:\Crimson_Sim\0D\Baseline\Base_Resting_DE_9.23_4.9_29_130_80_RpMin_Coronary_Healthy_CorrectPim"
    
    #BaseFile_Resting_DE_9.23_4.9_29_130_80_RpMin_Coronary_Healthy_CorrectPim_2sInjec
    # base_path = r"X:\Haizhou\Crimson_Sim\Patient117\3D\Hyper_Patient_Match\scalar_11_best"
    ### extract value
    
    xml_path = os.path.join(base_path, 'netlist_surfaces.xml')
    
    elastance_file_path = os.path.join(base_path, 'elastanceControl.py')
    
    matrix = extract_matrix_from_xml(xml_path)
    
    elastanceMatrix = extract_elastanceMatrix(elastance_file_path)
    
    LA_pressure=np.array(extract_LA_pressure(xml_path))
    
    
    initial_matrix = matrix.copy()
    initial_elastanceMatrix = elastanceMatrix.copy()
    initial_LA_pressure = LA_pressure.copy()

    ### modify
    
    # matrix = np.array([[0.005206897188488418, 0.15918831092429198 ,12.997010223417973, None, None],  # Aorta
    #                    [1e-05, 1e-05, 1e-05, 0.000273, 3e-06],  # Heart[L_AV,D_AV,L_MV,D_MV,T_LV resistance]
    #                    [6.2, 1.86, 17.325, 0.00072, 0.021],  # LM/LAD
    #                    [6.2, 1.86, 17.325, 0.00144, 0.04],  # OM1
    #                    [6.2, 1.86, 17.325,  0.00072, 0.021],  # OM2
    #                    [6.2, 1.86, 17.325, 0.00072, 0.021],  # LCx
    #                    [6.2, 1.86, 17.325, 0.00072, 0.021],  # AM
    #                    [6.2, 1.86, 17.325, 0.00102, 0.029]])  # RCA
    # #AM as unit 1 [6.2, 1.86, 17.325, 0.00072, 0.021] Area=0.9 mm^2
    # for i in (0.1,):
    #     matrix[2:8,1:3]*=0.26
    #     matrix[2:8,0]*=0.3
    #     matrix[2:8,3]*=0.8
    #     matrix[2:8,4]*=3
    
    for i in (1,):#range(7):#
    # # elastanceMatrix[3]*=4
    
    ############################################################################
        # resting_parameters
        # matrix[0,0]*=1.8
        # matrix[0,1]*=0.9
        # matrix[0,2]*=1.3
        # elastanceMatrix[3]*=0.87
        #     ### run main fuction 
        # LA_pressure*=1

    ############################################################################
        # resting_parameters
        # matrix[0,0]*=1.5
        # matrix[0,1]*=0.55
        # matrix[0,2]*=0.9
        # elastanceMatrix[3]*=1
        #     ### run main fuction 
        # LA_pressure*=1.2
    
    
    
    
    ############################################################################

        # matrix[0,0]*=1.3
        # matrix[0,1]*=0.5
        # matrix[0,2]*=1.15
    #     matrix[1,1]=1e-3
        
        # matrix[2, :3]*=1.4
        # matrix[3, :3]*=0.6
        # matrix[4, :3]*=1.15
        # matrix[5, :3]*=1.2
        # matrix[6, :3]*=0.73
        # matrix[7, :3]*=1.2
        
        # matrix[2, -2] *= 1
        # matrix[3, -2] *= 2
        # matrix[4, -2] *= 1
        # matrix[5, -2] *= 1
        # matrix[6, -2] *= 1
        # matrix[7, -2] *= 0.5
        
        # matrix[2, -1] *= 5
        # matrix[3, -1] *= 5
        # matrix[4, -1] *= 5
        # matrix[5, -1] *= 5
        # matrix[6, -1] *= 5
        # matrix[7, -1] *= 2
        
        # matrix[2, -2] = 0.0048*3
        # matrix[2, -1] = 0.027*5
        
        # matrix[3:6, -2] = 0.0048*1.5
        # matrix[3:6, -1] = 0.027*5
         
        
        # matrix[6, -2] = 0.0048*0.3
        # matrix[6, -1] = 0.027*7
        
        # matrix[7, -2] = 0.0048*5
        # matrix[7, -1] = 0.027*7
        
        # matrix[2:, :3]*=1.05-0.03*i
        # matrix[2:, :3]*=1.3
        # matrix[2:, -2:]*=0.95
        # matrix[2:, -1]*=10
        # matrix[2:, 2]*=1+2*i
        # matrix[6:, -2]*=0.1
        # matrix[6:, -1]*=100
        # matrix[6:, 3]*=10**(i-10)
        # matrix[6, -2]=0.0114
        # matrix[2:, :3]*=0.375
        
        matrix[2:, 0:2]*=2
        matrix[2:, 2:3]*=2
        matrix[2:, 3:]*=1
        result_dic,result_path = main_function(base_path, matrix, elastanceMatrix,LA_pressure,delete=False,heart_plot = "None",pvplot="All")
        LCA_flowrate = result_dic['outflow_LM_LAD']['mean_flow']+result_dic['outflow_OM1']['mean_flow']+result_dic['outflow_OM2']['mean_flow']+result_dic['outflow_LCx']['mean_flow']
        print(f'LCA flowrate is: {LCA_flowrate}')
        matrix = initial_matrix.copy()
        elastanceMatrix = initial_elastanceMatrix.copy()
        LA_pressure = initial_LA_pressure.copy()
        