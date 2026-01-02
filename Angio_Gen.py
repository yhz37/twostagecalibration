# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:12:21 2023

@author: haizhouy
"""

import sys
from sys import platform
import argparse
import cv2 
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.io import savemat
import os
import re

if platform == "win32":
    sys.path.insert(0, 'C:\\Program Files\\ParaView 5.11.0\\bin\\Lib\\site-packages_forpytorch')
elif platform == "linux":
    sys.path.insert(0,'/home/haizhouy/apt/ParaView-5.11.1/lib/python3.9/site-packages')


from paraview.simple import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)
plt.ion()
def get_time_step_size(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to find the time step size
    pattern = r'Time Step Size\s*:\s*([\d.]+)'

    # Search for the pattern in the content
    match = re.search(pattern, content)
    
    if match:
        time_step_size = float(match.group(1))
        return time_step_size
    else:
        raise ValueError("Time Step Size not found in the file")

def Angio_Gen(default_path = r'X:\Haizhou\Crimson_Sim\Patient117\3D\Resting_Patient_Match_correctPim\scalar_2_best\restarts-19_80_result-19000-80000-200',view = 'P17_LCA_-21.9_-18.3'):
    parser = argparse.ArgumentParser('Angio_Gen')
    parser.add_argument('--input_path', default=default_path, type=str)
    args = parser.parse_args()

    with open(os.path.join(args.input_path, 'view.pht'), 'r') as file:
        lines = file.readlines()
    line_to_check = '  <Field paraview_field_tag="iodine_contrast"\n'
    if line_to_check not in lines:
        new_lines = [
            '  <Field paraview_field_tag="iodine_contrast"\n',
            '         phasta_field_tag="solution"\n',
            '         start_index_in_phasta_array="5"\n',
            '         number_of_components="1"/>\n'
        ]
        lines[10]='<Fields number_of_fields="5">\n'
        lines[35:35] = new_lines

        # Open the file in write mode and write the updated content
        with open(os.path.join(args.input_path, 'view.pht'), 'w') as file:
            file.writelines(lines)
    
    pattern = r'result-(\d+)-(\d+)-(\d+)'
    # Search for the pattern in the path
    match = re.search(pattern, default_path)
    numbers = match.groups()
    time_steps = int(numbers[-1])
    time_step_size = get_time_step_size(os.path.join(args.input_path, 'solver.inp'))
    result_step_size = time_steps * time_step_size
    result_start_time_step = int(numbers[0])

    viewpht = PhastaReader(registrationName='view.pht', FileName=os.path.join(args.input_path, 'view.pht'))

    # viewpht = PhastaReader(registrationName='view.pht', FileName='D:\\Crimson_Sim\\scalar_9\\restarts-42_86_result-42000-86000-100\\view.pht')

    # get animation scene
    animationScene1 = GetAnimationScene()

    # get the time-keeper
    # timeKeeper1 = GetTimeKeeper()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    viewphtDisplay = Show(viewpht, renderView1, 'UnstructuredGridRepresentation')


    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # current camera placement for renderView1
    if view == 'RCA':
        renderView1.CameraPosition = [-55.21725875352469, -205.3587928487892, -157.11140353299027]
        renderView1.CameraFocalPoint = [-17.40080689557541, -45.52224779381188, -179.89075281944906]
        renderView1.CameraViewUp = [-0.5768046705155891, 0.24741956320413583, 0.7785113562532517]
        renderView1.CameraParallelScale = 62.83582041926124

    elif view == 'P17_LCA_-21.9_-18.3':
        
        # before smoothing coronary outlet
        # renderView1.CameraPosition = [-61.937097788648806, -602.6877674797187, -250.89723477347604]
        # renderView1.CameraFocalPoint = [7.88285002274333, -180.9793065907642, -145.94908189397938]
        # renderView1.CameraViewUp = [0.1255218443008741, -0.259108548471095, 0.9576570506775949]
        # renderView1.CameraViewAngle = 18.049327354260086
        # renderView1.CameraParallelScale = 77.80733282339307
        
        # after smoothing coronary outlet
        renderView1.CameraPosition = [3.6179235166102215, -367.938900917586, -204.20886116449864]
        renderView1.CameraFocalPoint = [14.984025761379954, -180.23608308028304, -149.59379152287468]
        renderView1.CameraViewUp = [0.11406805259590265, -0.2839179975130046, 0.9520394162350548]
        renderView1.CameraParallelScale = 74.20234247302223
    
    elif view == 'P17_LCA_0.2_-35.2':
        # before smoothing coronary outlet
        # renderView1.CameraPosition = [36.610970777884475, -425.51710768543603, -412.68626370641306]
        # renderView1.CameraFocalPoint = [20.902221693766435, -168.87316725293877, -155.3824599806475]
        # renderView1.CameraViewUp = [-0.0013218616658655753, -0.7080537908657824, 0.7061572642990593]
        # renderView1.CameraViewAngle = 18.049327354260086
        # renderView1.CameraParallelScale = 77.80733282339307
        # after smoothing coronary outlet
        renderView1.CameraPosition = [28.833658297254456, -321.5850037080369, -278.66316950495934]
        renderView1.CameraFocalPoint = [13.24497637641099, -170.58755690065738, -154.96553011808774]
        renderView1.CameraViewUp = [0.1057476266719391, -0.6236022063885974, 0.7745564715632589]
        renderView1.CameraParallelScale = 74.20234247302223


    # set scalar coloring
    ColorBy(viewphtDisplay, ('POINTS', 'iodine_contrast'))

    # rescale color and/or opacity maps used to include current data range
    viewphtDisplay.RescaleTransferFunctionToDataRange(True, False)


    # get color transfer function/color map for 'iodine_contrast'
    iodine_contrastLUT = GetColorTransferFunction('iodine_contrast')


    # change representation type
    viewphtDisplay.SetRepresentationType('Volume')
    # viewphtDisplay.PointSize = 0.1
    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    iodine_contrastLUT.ApplyPreset('X Ray', True)

    # get opacity transfer function/opacity map for 'iodine_contrast'
    iodine_contrastPWF = GetOpacityTransferFunction('iodine_contrast')
    
    # Rescale transfer function
    iodine_contrastLUT.RescaleTransferFunction(0.0, 400.0)

    # Rescale transfer function
    iodine_contrastPWF.RescaleTransferFunction(0.0, 400.0)

    # get 2D transfer function for 'iodine_contrast'
    iodine_contrastTF2D = GetTransferFunction2D('iodine_contrast')

    # Rescale 2D transfer function
    iodine_contrastTF2D.RescaleTransferFunction(0.0, 400.0, 0.0, 1.0)
    
    # hide color bar/color legend
    viewphtDisplay.SetScalarBarVisibility(renderView1, False)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1174, 526)

    try:
        os.makedirs(os.path.join(args.input_path, 'Angio_'+view))
    except:
        pass

    # save animation
    SaveAnimation(os.path.join(args.input_path, 'Angio_'+view, 'Angio.jpeg'), renderView1, ImageResolution=[526, 526],
        OverrideColorPalette='WhiteBackground',
        FrameWindow=[0, 440],
        Quality=100)
    Delete(viewpht)
    del viewpht
    Delete(renderView1)
    del renderView1
    Delete(animationScene1)
    del animationScene1
    Delete(viewphtDisplay)
    del viewphtDisplay
    Delete(iodine_contrastLUT)
    del iodine_contrastLUT
    Delete(iodine_contrastPWF)
    del iodine_contrastPWF
    Delete(layout1)
    del layout1

    ################# Binary#################

    # Get the height and width of the images
    sample_image = cv2.imread(os.path.join(args.input_path, 'Angio_'+view, 'Angio.0000.jpeg'), cv2.IMREAD_GRAYSCALE)
    height, width = sample_image.shape

    # Define the codec and create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec as needed
    video_writer = cv2.VideoWriter(os.path.join(args.input_path, 'Angio_'+view, 'binary.mp4'), fourcc, int(1/result_step_size), (width, height))
    xray_writer = cv2.VideoWriter(os.path.join(args.input_path, 'Angio_'+view, 'x-ray.mp4'), fourcc, int(1/result_step_size), (width, height))


    threshold_value = 245
    white_pixel_counts = []

    all_files = glob.glob(os.path.join(args.input_path, 'Angio_'+view, '*.jpeg'))
    filtered_files = [file for file in all_files if not file.endswith('_binary.jpeg')]

    
    for filename in sorted(filtered_files):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        white_pixel_counts.append(cv2.countNonZero(binary_image))
        cv2.imwrite( filename[:-5] + '_binary.jpeg', binary_image)
        video_writer.write(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR))
        xray_writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        
    # Release the video writer
    video_writer.release()
    xray_writer.release()
        
    white_pixel_counts = np.asarray(white_pixel_counts)
    x = np.linspace(0, (white_pixel_counts.shape[0]-1)*result_step_size, num=white_pixel_counts.shape[0])+result_start_time_step*time_step_size
    plt.ioff()
    fig = plt.figure()
    plt.plot(x,white_pixel_counts/np.max(white_pixel_counts),linewidth=2,color = 'black')
    plt.xlabel("Time (s)",fontsize=28,weight='bold')
    plt.ylabel("Normalized Pixel Count",fontsize=28,weight='bold')
    plt.xticks(fontsize=20,weight='bold')
    plt.yticks(fontsize=20,weight='bold')
    fig.subplots_adjust(bottom=0.17)
    fig.subplots_adjust(left=0.2)
    plt.ticklabel_format(useOffset=False)
    resolution_value = 1200
    plt.savefig(os.path.join(args.input_path, 'Angio_'+view, 'Intensity.png'), format="png", dpi=resolution_value)
    # plt.show()
    plt.close(fig)

    savemat(os.path.join(args.input_path, 'Angio_'+view, 'white_pixel_count_data.mat'), {'white_pixel_counts': white_pixel_counts,'result_step_size':result_step_size})

    return white_pixel_counts,result_step_size

if __name__ == "__main__":
    white_pixel_counts,result_step_size = Angio_Gen()

