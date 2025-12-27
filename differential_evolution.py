import os
import numpy as np
from scipy.optimize import differential_evolution
from main_function import main_function,extract_matrix_from_xml,extract_elastanceMatrix,extract_LA_pressure

import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
from sys import platform

case = 'Resting'

if case ==  'Hyperimia':
    if platform == "win32":
        base_path = r"D:\Crimson_Sim\0D\Baseline\Base_Hyper_y"
    elif platform == "linux":
        base_path = '/nfs/turbo/umms-figueroc/Haizhou/Crimson_Sim/Patient117/0D/Parameter_Tuning/Base_Hyper_y'

elif case =='Resting':
    if platform == "win32":
        base_path = r"D:\Crimson_Sim\0D\Baseline\Base_Resting_DE_9.23_4.9_29_130_80_RpMin_Coronary_Healthy"
    elif platform == "linux":
        base_path = '/nfs/turbo/umms-figueroc/Haizhou/Crimson_Sim/Patient117/0D/Parameter_Tuning/Base_Resting_DE_9.23_4.9_29_130_80_RpMin_Coronary_Healthy'

xml_path = os.path.join(base_path, 'netlist_surfaces.xml')
elastance_file_path = os.path.join(base_path, 'elastanceControl.py')
matrix_initial = extract_matrix_from_xml(xml_path)
elastanceMatrix_initial = extract_elastanceMatrix(elastance_file_path)
LA_pressure_initial= np.array(extract_LA_pressure(xml_path))
            


def objective_function(x):

    matrix = matrix_initial.copy()
    elastanceMatrix = elastanceMatrix_initial.copy()
    LA_pressure = LA_pressure_initial.copy()
    
    matrix[0,0] = x[0]
    matrix[0,1] = x[1]
    matrix[0,2] = x[2]
    elastanceMatrix[3] = x[3]
    LA_pressure = x[4]
    elastanceMatrix[2] = x[5]
    elastanceMatrix[0] = x[6]

    
    result_dic,result_path = main_function(base_path, matrix, elastanceMatrix,LA_pressure,delete=True,heart_plot = None,pvplot = None)



    inflow_Aorta = result_dic['inflow_Aorta']


    LV = result_dic['LV']
    # print(f"End-diastolic volume :{LV['max_V']}")
    
    ################## Loss_Resting ###############################
    if case =='Resting':
        loss = 2*abs((inflow_Aorta['mean_flow'] - 4.9)/4.9) +\
          abs((inflow_Aorta['max_pressure']-130)/130)+abs((inflow_Aorta['min_pressure']-80)/80)+\
            2*abs((inflow_Aorta['max_flow']-28)/28)
    elif case ==  'Hyperimia':
        loss = abs((inflow_Aorta['mean_flow'] - 8.8)/8.8) +\
              2*abs((inflow_Aorta['max_pressure']-125)/125)+abs((inflow_Aorta['min_pressure']-75)/75)+\
                0.5*abs((inflow_Aorta['max_flow']-38)/38)
    
    pulse_pressure = inflow_Aorta['max_pressure'] - inflow_Aorta['min_pressure']
    DNP_lb = inflow_Aorta['max_pressure'] -0.5*pulse_pressure
    DNP_ub = inflow_Aorta['max_pressure'] -0.2*pulse_pressure
    DNP_medium = inflow_Aorta['max_pressure'] -0.35*pulse_pressure
    if inflow_Aorta['dicrotic_notch_pressure']<DNP_lb:
        loss += (DNP_lb-inflow_Aorta['dicrotic_notch_pressure'])/DNP_medium
    elif inflow_Aorta['dicrotic_notch_pressure']>DNP_ub:
        loss += (inflow_Aorta['dicrotic_notch_pressure']-DNP_ub)/DNP_medium
    
    if case =='Resting':
        EDV_lb = 100
        EDV_ub = 160
        EDV_medium = 130
    elif case ==  'Hyperimia':
        EDV_lb = 100
        EDV_ub = 170
        EDV_medium = 130
        

    if LV['max_V']<EDV_lb:
        loss += (EDV_lb-LV['max_V'])/EDV_medium
    elif LV['max_V']>EDV_ub:
        loss += (LV['max_V']-DNP_ub)/EDV_medium
    
    ################## Loss_Hyper ###############################


    print(f'loss :{loss}')

    return loss

#callback function
def callback(xk, convergence):
    loss = objective_function(xk)
    current_best = xk.tolist()
    history.append((current_best, loss))
    print(f'Best design is: {xk}')
    print(f'Current best loss: {loss}')


    
    
if __name__ == "__main__":

    matrix = matrix_initial.copy()
    elastanceMatrix = elastanceMatrix_initial.copy()
    LA_pressure = LA_pressure_initial.copy()

    if case ==  'Resting':
        bounds = [(0.005*0.6,0.005*3), (0.15*0.5, 0.15*2), (15-10, 15+15), (1/7.501, 2/7.501), (1500, 2300), (0.0053, 0.016),
                  (0.35, 0.39)]
    elif case =='Hyperimia':
        data = loadmat( os.path.join("result", "Base_Resting_DE_9.23_4.9_29_130_80.mat"))
        x = data['solution_dv']

        bounds = [(0.3*x[0,0],0.9*x[0,0]), (0.3*x[0,1], 0.9*x[0,1]), (x[0,2]-15, x[0,2]-2), (x[0,3]*1.2, 4/7.501), (x[0,4]*1.02, x[0,4]*1.08), (x[0,5], x[0,5]),
                  (x[0,6]*0.95, x[0,6]*1.05)]
    

    history = []
    
    # Start DE OP
    result = differential_evolution(
        objective_function,
        bounds,
        popsize=7,
        maxiter=150,
        polish=False,
        disp=True,
        workers=10,
        callback=callback
    )

    

    # Extract the losses for plotting
    _, losses = zip(*history)
    
    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title('Optimization Loss Over Generations')
    plt.grid(True)
    
    # Save the figure
    figure_file_path = os.path.join(os.path.dirname(base_path), 'optimization_history.png')
    plt.savefig(figure_file_path)  # Save the figure to a file
    
    solution_dv = result.x
    solution_fitness = result.fun
    plt.show()  # Show the figure
    if case ==  'Resting':
        savemat("OP_results_DE_rest.mat",{'base_path':base_path,'solution_dv':solution_dv,'solution_fitness':solution_fitness,'losses':losses})

    elif case =='Hyperimia':
        savemat("OP_results_DE_hyper.mat",{'base_path':base_path,'solution_dv':solution_dv,'solution_fitness':solution_fitness,'losses':losses})
