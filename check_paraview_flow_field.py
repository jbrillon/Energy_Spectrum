import numpy as np
from Energy_Spectrum import compute_Ek_spectrum
import matplotlib.pyplot as plt

import os;CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]+"/";
import sys; sys.path.append(CURRENT_PATH+"../PHiLiP-Post-Processing/src/tools");
from assemble_mpi_flow_field_files_and_reorder import assemble_mpi_flow_field_files_and_reorder

# load PHiLiP velocity field
#-----------------------------------------------------
from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    filesystem="/media/julien/Samsung_T5/"
elif platform == "darwin":
    # OS X
    filesystem="/Volumes/Samsung_T5/"

# assemble_mpi_flow_field_files_and_reorder(
#     "/home/julien/NarvalFiles/2023_JCP/filtered_dns_viscous_tgv/viscous_TGV_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0256_p7_procs1024/solution_files/velocity_field_extracted_from_paraview","txt",1024,6,32,7)
# exit()

nRows=256*256*256

path = "NarvalFiles/2023_JCP/filtered_dns_viscous_tgv/viscous_TGV_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0256_p7_procs1024/"
file = filesystem+path+"flow_field_files/velocity_vorticity-0_reordered.dat"
data = np.loadtxt(file,skiprows=1,usecols=(0,1,2,3,4,5),dtype=np.float64)
# file_original = np.array([data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]],dtype=np.float64) # x,y,z,u,v,w
file_original = [data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]] # x,y,z,u,v,w
file_original = np.zeros((nRows,6))
for i in range(0,6):
    file_original[:,i] = data[:,i] 

# load the 256^3 P7
path = "NarvalFiles/2023_JCP/filtered_dns_viscous_tgv/viscous_TGV_ILES_NSFR_cDG_IR_2PF_GL_OI-0_dofs0256_p7_procs1024/"
file = "/home/julien/"+path+"solution_files/velocity_field_extracted_from_paraview_reordered.txt"
data = np.loadtxt(file,skiprows=1,usecols=(0,1,2,3,4,5),delimiter=" ",dtype=np.float64)
file_new = np.zeros((nRows,6))
for i in range(0,3):
    file_new[:,i] = data[:,i+3] 
for i in range(0,3):
    file_new[:,i+3] = data[:,i]

for i in range(0,nRows):
    err = np.abs(np.linalg.norm(file_original[i,:]-file_new[i,:]))
    if(err>3.0e-7):
        print("Error: %1.6e" % err)
        print("Line number: %i" % i)
        print(file_original[i,:])
        print(file_new[i,:])
        print("Aborting...")
        exit()
print("Completed. Files are within the tolerance.")