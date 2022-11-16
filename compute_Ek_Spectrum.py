# python 3

# Script for the computation of 3D spectrum of the Total Kinetic Energy
# Adapted to the Taylor-Green vortex (TGV) problem.
# CREATED by FARSHAD NAVAH
# McGill University
# farshad.navah .a.t. mail.mcgill.ca
# 2018
# provided as is with no garantee.
# Please cite:
#    https://github.com/fanav/Energy_Spectrum
#    https://arxiv.org/abs/1809.03966

# -----------------------------------------------------------------
#  IMPORTS - ENVIRONMENT
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt

def compute_Ek_spectrum(velocity_field=[],reference_velocict=1.0):
    U0 = 1.0*reference_velocicty # to non-dimensionalize the inputted velocity field
    
    #-----------------------------------------------------
    # Safeguard for when empty data is passed
    #-----------------------------------------------------
    if(velocity_field==[]):
        print("compute_Ek_spectrum error: velocity_field is empty")
        print("aborting...")
        return

    # Load velocity components
    U,V,W = velocity_field # must be ordered as x,y,z
    # # -----------------------------------------------------------------
    # #  OUTOUT FILE PARAMS
    # # -----------------------------------------------------------------

    # Figs_Path = "./"
    # Fig_file_name = "Ek_Spectrum"

    # -----------------------------------------------------------------
    #  READ FILES
    # -----------------------------------------------------------------

    # localtime = time.asctime( time.localtime(time.time()) )
    # print ("\nReading files...localtime",localtime)

    # #load the ascii file
    # if(file == "velocityfld_ascii.dat"):
    #     n_skiprows = 2
    # else:
    #     n_skiprows = 0
    # data = np.loadtxt(data_path+file, skiprows=n_skiprows)

    print ("shape of data = ",data.shape)

    # localtime = time.asctime( time.localtime(time.time()) )
    # print ("Reading files...localtime",localtime, "- END\n")

    # -----------------------------------------------------------------
    #  COMPUTATIONS
    # -----------------------------------------------------------------
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime)

    N = int(round((len(U)**(1./3))))
    print("N =",N)
    eps = 1e-50 # to void log(0)

    U = U.reshape(N,N,N)/U0
    V = V.reshape(N,N,N)/U0
    W = W.reshape(N,N,N)/U0

    amplsU = abs(np.fft.fftn(U)/U.size)
    amplsV = abs(np.fft.fftn(V)/V.size)
    amplsW = abs(np.fft.fftn(W)/W.size)

    EK_U  = amplsU**2
    EK_V  = amplsV**2 
    EK_W  = amplsW**2 

    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)
    EK_W = np.fft.fftshift(EK_W)

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]
    sign_sizez = np.shape(EK_U)[2]

    box_sidex = sign_sizex
    box_sidey = sign_sizey
    box_sidez = sign_sizez

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2+(box_sidez)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
    centerz = int(box_sidez/2)

    print ("box sidex     =",box_sidex) 
    print ("box sidey     =",box_sidey) 
    print ("box sidez     =",box_sidez)
    print ("sphere radius =",box_radius )
    print ("centerbox     =",centerx)
    print ("centerboy     =",centery)
    print ("centerboz     =",centerz,"\n" )
                    
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    EK_W_avsphr = np.zeros(box_radius,)+eps ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            for k in range(box_sidez):            
                wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2+(k-centerz)**2)))
                EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j,k]
                EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j,k]    
                EK_W_avsphr[wn] = EK_W_avsphr [wn] + EK_W [i,j,k]        

    EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)

    realsize = len(np.fft.rfft(U[:,0,0]))
    # plt.loglog(np.arange(0,realsize),((EK_avsphr[0:realsize] )),'k-.')
    # plt.loglog(np.arange(realsize,len(EK_avsphr),1),((EK_avsphr[realsize:] )),'r--')


    # axes = plt.gca()
    # axes.set_ylim([10**-25,5**-1])
    # if(file == "velocityfld_ascii.dat"):
        # axes.set_ylim([1e-8,1e-1])
        # axes.set_xlim([2,80])
    # else:
        # axes.set_ylim([1e-4,2e-1])
        # axes.set_xlim([1e0,1e2])

    print("Real      Kmax    = ",realsize)
    print("Spherical Kmax    = ",len(EK_avsphr))

    TKEofmean_discrete = 0.5*(np.sum(U/U.size)**2+np.sum(V/V.size)**2+np.sum(W/W.size)**2)
    TKEofmean_sphere   = EK_avsphr[0]

    total_TKE_discrete = np.sum(0.5*(U**2+V**2+W**2))/(N*1.0)**3
    total_TKE_sphere   = np.sum(EK_avsphr)

    print("the KE  of the mean velocity discrete  = ",TKEofmean_discrete)
    print("the KE  of the mean velocity sphere    = ",TKEofmean_sphere )
    print("the mean KE discrete  = ",total_TKE_discrete)
    print("the mean KE sphere    = ",total_TKE_sphere)

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime, "- END \n")

    # -----------------------------------------------------------------
    #  OUTPUT/PLOTS
    # -----------------------------------------------------------------

    # dataout      = np.zeros((box_radius,2)) 
    # dataout[:,0] = np.arange(0,len(dataout))
    # dataout[:,1] = EK_avsphr[0:len(dataout)]
    computed_spectra_from_velocity = np.zeros((realsize,2))
    computed_spectra_from_velocity[:,0] = np.arange(0,realsize)
    computed_spectra_from_velocity[:,1] = ((EK_avsphr[0:realsize] ))

    return computed_spectra_from_velocity
    # np.savetxt("computed_spectra.txt",dataout)

    # np.savetxt(Figs_Path+Fig_file_name+'.dat',dataout)
    # fig.savefig(Figs_Path+Fig_file_name+'.pdf')
    # plt.show()
