# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:14:01 2025

@author: haizhouy
"""

from scipy.io import savemat,loadmat
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

plt.rcParams["font.family"] = "Arial"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)
plt.ion()

def numerical_sort(folder_name):
    return int(folder_name.split('_')[1])

def find_folder_with_substring(substring, search_directory):
    items_in_directory = os.listdir(search_directory)
    for item in items_in_directory:
        if substring in item and os.path.isdir(os.path.join(search_directory, item)):
            return os.path.join(search_directory, item)
    return None

def find_folders_with_exact_substring(root_folder, substring):
    
    # List all files and folders in the specified directory
    all_files = os.listdir(root_folder)
    folder_list = [folder for folder in all_files if os.path.isdir(os.path.join(root_folder, folder)) and folder.startswith(substring)]
    folder_list.sort(key=numerical_sort)
    matching_folders = [os.path.abspath(os.path.join(root_folder, folder)) for folder in folder_list if os.path.isdir(os.path.join(root_folder, folder)) and folder.startswith(substring)]
    return matching_folders

def find_increase_start(curve):
    for i in range(1, len(curve)):
        if curve[i] > curve[i-1]:
            return i-1
    return -1  # If no increase is found

def get_LCA_flowrate(vp_dics):
    LAD_flows = []
    for idx in range(vp_dics.shape[1]):
        vp_dic = vp_dics[0,idx][0,0]
        LAD_flow = vp_dic['outflow_LM_LAD'][0,0]['mean_flow'][0,0]
        LAD_flows.append(LAD_flow)
        
    return np.array(LAD_flows)

def MSE_compute(Patient_Data,pcs,result_step_sizes,case):
    MSEs = []

    for idx in range(pcs.shape[0]):
        ts_s = result_step_sizes[idx].item()
        CIP_s = pcs[idx].flatten()

        ts_p = Patient_Data['Frame_Time'].item()/1000
        CIP_p = Patient_Data['white_pixel_counts'].flatten()

        start_s = find_increase_start(CIP_s)
        start_p = find_increase_start(CIP_p)
        
        if case == 'Hyperemia':
            start_p = 34
        CIP_s = CIP_s[start_s:]
        CIP_p = CIP_p[start_p:]


        x_s = np.linspace(0, (CIP_s.shape[0]-1)*ts_s, num=CIP_s.shape[0])
        x_p = np.linspace(0, (CIP_p.shape[0]-1)*ts_p, num=CIP_p.shape[0])
        
        
        normalized_CIP_p_inter = np.interp(x_p, x_s[:], CIP_s/np.max(CIP_s))
        MSE = np.mean((normalized_CIP_p_inter-CIP_p/np.max(CIP_p))**2)
        MSEs.append(MSE)
        # fig = plt.figure()
        # plt.plot(x_s,CIP_s/np.max(CIP_s),'#1F77B4',linewidth=2, label='Simulation')
        # plt.plot(x_p,CIP_p/np.max(CIP_p),'#FF7F0E',linestyle = '--',linewidth=2, label='Clinical')


        # plt.xlabel("Time (s)",fontsize=24,weight='bold')
        # plt.ylabel("Normalized Pixel Count",fontsize=24,weight='bold')
        # # plt.xlim([0,min((CIP_s.shape[0]-1)*ts_s,(CIP_p.shape[0]-1)*ts_p)])
        # plt.xlim([0,4])
        # plt.xticks(fontsize=20,weight='bold')
        # plt.yticks(fontsize=20,weight='bold')
        # fig.subplots_adjust(bottom=0.17)
        # fig.subplots_adjust(left=0.2)
        # plt.locator_params(axis='x',nbins=5)
        # plt.ticklabel_format(useOffset=False)
        # resolution_value = 1200
        # plt.show()
        # # plt.legend(shadow=True,prop={'weight':'bold','size':16})
        # plt.savefig('CIP_comparison_{}_{}.png'.format(idx+1,case), format="png", dpi=resolution_value) 
    return np.array(MSEs)

patient_data_path_rest = r"D:\OneDrive - Michigan Medicine\Research\IMR\Patient Data\Angios\017_Angio de-id\IMR24_ANGIO_017-3_anonymized_(-21.9,-18.3)\white_pixel_count_data_fix_REV1.mat"
Sim_Data_path_rest = r'X:\Haizhou\Crimson_Sim\Patient117\3D\Resting_Patient_Match_correctPim\post_data.mat'
patient_data_path_hyper = r"D:\OneDrive - Michigan Medicine\Research\IMR\Patient Data\Angios\017_Angio de-id\IMR24_ANGIO_017-7_anonymized_(0.2,-35.2)\white_pixel_count_data.mat"
Sim_Data_path_hyper = r'X:\Haizhou\Crimson_Sim\Patient117\3D\Hyper_Patient_Match_correctPim\post_data.mat'
####################################### Rest
Sim_Data_rest = loadmat(Sim_Data_path_rest)
vp_dics_rest = Sim_Data_rest['vp_dics']
pcs_rest = Sim_Data_rest['pcs']
result_step_sizes_rest = Sim_Data_rest['result_step_sizes']
LAD_flows_rest = get_LCA_flowrate(vp_dics_rest)

Patient_Data_rest = loadmat(patient_data_path_rest)
mse_rest =  MSE_compute(Patient_Data_rest,pcs_rest,result_step_sizes_rest,'rest')


######################################### hyper
Sim_Data_hyper = loadmat(Sim_Data_path_hyper)
vp_dics_hyper = Sim_Data_hyper['vp_dics']
pcs_hyper = Sim_Data_hyper['pcs']
result_step_sizes_hyper = Sim_Data_hyper['result_step_sizes']

LAD_flows_hyper = get_LCA_flowrate(vp_dics_hyper)


Patient_Data_hyper = loadmat(patient_data_path_hyper)
mse_hyper =  MSE_compute(Patient_Data_hyper,pcs_hyper,result_step_sizes_hyper,'Hyperemia')


obj_fun_val = np.zeros([mse_rest.shape[0],mse_hyper.shape[0]])
for ii in range(mse_rest.shape[0]):
    for jj in range(mse_hyper.shape[0]):
        obj_fun_val[ii,jj] = mse_rest[-ii-1]+mse_hyper[jj]+abs(LAD_flows_hyper[jj]/LAD_flows_rest[-ii-1]-2.2)/2.2
        
# Create a figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(7.5, 5.5)
# Generate the heatmap
cax = ax.imshow(obj_fun_val, cmap='viridis', aspect='auto')

# Add a color bar to indicate the value scale
colorbar = fig.colorbar(cax, ax=ax)
colorbar.set_label('Objective Function Value', fontsize=16)  # Larger label font size
colorbar.ax.tick_params(labelsize=14)

for i in range(obj_fun_val.shape[0]):
    for j in range(obj_fun_val.shape[1]):
        ax.text(j, i, f'{obj_fun_val[i, j]:.2f}', ha='center', va='center', color='white',fontdict={'family': 'Arial', 'size': 16})



# # Custom labels
# x_labels = [
#     '$91\%\ R_{T,i}^{r,pre}$',
#     '$94\%\ R_{T,i}^{r,pre}$',
#     '$97\%\ R_{T,i}^{r,pre}$',
#     '$R_{T,i}^{r,pre}$',
#     '$103\%\ R_{T,i}^{r,pre}$',
#     '$106\%\ R_{T,i}^{r,pre}$',
#     '$109\%\ R_{T,i}^{r,pre}$'
# ]

x_labels = [
    r'-$9\%\mathbf{H}$',
    r'-$6\%\mathbf{H}$',
    r'-$3\%\mathbf{H}$',
    r'$\mathbf{H}$',
    r'$3\%\mathbf{H}$',
    r'$6\%\mathbf{H}$',
    r'$9\%\mathbf{H}$'
]

y_labels = [
    r'-$9\%\mathbf{R}$',
    r'-$6\%\mathbf{R}$',
    r'-$3\%\mathbf{R}$',
    r'$\mathbf{R}$',
    r'$3\%\mathbf{R}$',
    r'$6\%\mathbf{R}$',
    r'$9\%\mathbf{R}$'
]
# Set tick positions and labels
ax.set_xticks(range(len(x_labels)))
ax.set_yticks(range(len(y_labels)))
ax.set_xticklabels(x_labels, fontsize=16)
ax.set_yticklabels(y_labels, fontsize=16)
# Show the plot
plt.show()
resolution_value = 1200
plt.savefig("fitness_table.jpg", format="jpg", dpi=resolution_value)            