"""
Modified on Dec 29 2025
"""

import numpy as np
from numpy import exp, linspace, random
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.signal import savgol_filter
from scipy.ndimage import convolve1d
from scipy.ndimage import correlate1d
from scipy.interpolate import interp1d

#============Get the current directory
path = os.getcwd()
print("The current directory is: %s" % path )
#==================================================================



#specify from which file the data comes
fileNameTemp = ['MBE_Ge_D6_-196C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-180C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-160C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-140C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-120C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-100C_frostRemoved_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-80C_frostRemoved_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-60C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-40C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_-20C_1064 nm_x50_100%_1000_300 gr_mm_Acc2_Acq3.txt',
                'MBE_Ge_D6_24C_1064 nm_x50_100%_1000_300 gr_mm_Acc3_Acq3.txt']
temperatureExp = [77, 93, 113, 133, 153, 173, 193, 213, 233, 253, 297]


def gaussian(x, amp, cen, wid, y0=0.0, normalized=False):
    if normalized == False:
        return y0 + np.divide(amp * np.exp(-(4*np.log(2))*np.power((x-cen),2) / np.power(wid,2)), wid*np.sqrt(np.divide(np.pi,(4*np.log(2)))))
    elif normalized == True:
        temp=y0 + np.divide(amp * np.exp(-(4*np.log(2))*np.power((x-cen),2) / np.power(wid,2)), wid*np.sqrt(np.divide(np.pi,(4*np.log(2)))))
        return np.divide(temp,np.amax(temp))

def biGaussian(x, height, cen, c1, c2, y0=0.0, normalized=False):
    #first calculate the lower side of the distribution
    output = y0 + height*np.exp(-np.power((x - cen),2)/(2*np.power(c1,2)))
    #Then find out to which element cen is closest to.
    closestElement=(np.abs(x - cen)).argmin()
    if (x[closestElement] - cen) >=0:
        closestElement=closestElement-1
    #Calculate all the values on the higher side of the distribution    
    output[closestElement:] = y0 + height*np.exp(-np.power((x[closestElement:] - cen),2)/(2*np.power(c2,2)))
    if normalized == False:
        return output
    elif normalized == True:
        return np.divide(output,np.amax(output))

def simpleDOS(E, Eg, A, normalized=True):
    if np.isscalar(E) == False:
        DOS=np.zeros(int(np.size(E)))
        n=0
        for i in E:
            if i<=Eg:
                DOS[n]=0.0
            else:
                DOS[n]=np.multiply(A,np.sqrt(i-Eg))
                
            n+=1
        return DOS     

    else:
        return np.multiply(A,np.sqrt(E-Eg))

def fermiDirac(E,Eg,T,Eshift=0):
    k=8.61733e-5 #[eV/K]
    return np.divide(1,(1+np.exp(np.divide((E-Eg-Eshift),(k*T)))))

def urbachTail(E, Eg, gam, theta=1, normalized=False):
    N=np.divide(1,(2*gam*gamma(1+np.divide(1,theta)))) #This normalization factor makes that the area under neath the graph remains to be 1
    if normalized == False:
        return np.multiply(np.exp(-np.power(np.absolute((np.divide((E-Eg),gam))),theta)),N)
    elif normalized == True:
        return np.divide(np.multiply(np.exp(-np.power(np.absolute((np.divide((E-Eg),gam))),theta)),N), np.amax(np.multiply(np.exp(-np.power(np.absolute((np.divide((E-Eg),gam))),theta)),N)))
    
def tailedDOS(E, Eg, A, gam, theta=1, normalized=False):
    #First make an array that will be used to define the urback tail, important is that I know the length of the array and that the peak is at te exact center
    EStepsize=np.divide((E[40]-E[0]),40) #Get the energy stepsize
    HalfLength = int(1000) #Choose a value for the length of the array used for the peak function 
    EDataUT = np.zeros(HalfLength*2) #define the array that will be used and fill the array with energy values with zero at the center of the array.
    for i in range (HalfLength):
        EDataUT[i]=(i-HalfLength+0.5)*EStepsize
        EDataUT[i+HalfLength]=(i+0.5)*EStepsize
    #Now the FULL convolution can be done between the urbach tail and the simple DOS    
    tailDOS=np.convolve(urbachTail(EDataUT, 0, gam, theta), simpleDOS(E, Eg, A), 'full')/sum(urbachTail(EDataUT, 0, gam, theta))
    #The convolution is now exactly shifted by half of the urbach tail length minus one cell So now I correct for this and we can return the array.
    tailDOS=tailDOS[HalfLength-1:]
    tailDOS=tailDOS[0:int(np.size(E))]
    #Then return the signal
    if normalized == False:
        return tailDOS
    elif normalized == True:
        return np.divide(tailDOS,np.amax(tailDOS))
    
def simpleDOS_FD(E, Eg, A, T, Eshift=0, normalized=False):
    if normalized == False:
        return np.multiply(simpleDOS(E,Eg,A),fermiDirac(E,Eg,T, Eshift))
    elif normalized == True:
        return np.divide(np.multiply(simpleDOS(E,Eg,A),fermiDirac(E,Eg,T, Eshift)),np.amax(np.multiply(simpleDOS(E,Eg,A),fermiDirac(E,Eg,T, Eshift))))

def tailedDOS_FD(E, Eg, A, T, gam, theta=1, Eshift=0.0, normalized=False):
    if normalized == False:
        return np.multiply(tailedDOS(E, Eg, A, gam, theta),fermiDirac(E,Eg,T,Eshift))
    elif normalized == True:
        return np.divide(np.multiply(tailedDOS(E, Eg, A, gam, theta),fermiDirac(E,Eg,T,Eshift)),np.amax(np.multiply(tailedDOS(E, Eg, A, gam, theta),fermiDirac(E,Eg,T,Eshift))))

def tailedAbsorptivity (E, Eg, A, gam, theta=1, Eshift=0, d=1, normalized=False):
    #d is a characeristic length scale of absorption
    if normalized == False:
        return 1-np.exp(-tailedDOS(E, Eg, A, gam, theta)*d)
    elif normalized == True:
        return np.divide((1-np.exp(-tailedDOS(E, Eg, A, gam, theta)*d)),np.amax(1-np.exp(-tailedDOS(E, Eg, A, gam, theta)*d)))
    
def simpleAbsorptivity (E, Eg, A, Eshift=0, d=1, normalized=False):
    #d is a characeristic length scale of absorption in paper used as fitting parameter
    if normalized == False:
        return 1-np.exp(-simpleDOS(E, Eg, A)*d)
    elif normalized == True:
        return np.divide((1-np.exp(-simpleDOS(E, Eg, A)*d)),np.amax(1-np.exp(-simpleDOS(E, Eg, A)*d)))
    
def tailedAbsortivityOccCorr(E, Eg, A, T, gam, Dmu, d=1, theta=1, Eshift=0, normalized=False):
    k=8.61733e-5 #[eV/K]
    #d is a characeristic length scale of absorption in paper used as fitting parameter
    temp=np.multiply(tailedDOS(E, Eg, A, gam, theta),(1-np.divide(2,np.exp(np.divide((E-Dmu),(2*np.multiply(k,T))))+1)))
    temp=1-np.exp(-temp*d)
    #temp=(1-np.divide(2,np.exp(np.divide((E-Dmu),(2*np.multiply(k,T))))+1))
    if normalized == False:
        return temp
    elif normalized == True:
        return np.divide(temp,temp[-1])
    
    
def LSW_tailedAbs_PL (E, Eg, A, T, gam, Dmu, d=1, theta=1, Eshift=0, normalized=False):
    k=8.61733e-5 #[eV/K]
    temp=np.divide(np.multiply(np.power(E,2),tailedAbsortivityOccCorr(E, Eg, A, T, gam, Dmu, d, theta, Eshift=0.0, normalized=False)),(np.exp(np.divide((E-Dmu),(np.multiply(k,T))))-1))
    #temp=np.divide(1,np.exp(np.divide((E-Dmu),(np.multiply(k,T))))-1)
    if normalized == False:
        return temp
    elif normalized == True:
        return np.divide(temp,np.amax(temp))
    
def LSW_tailedAbsNoCorr_PL (E, Eg, A, T, gam, Dmu, d=1, theta=1, Eshift=0, normalized=False):
    k=8.61733e-5 #[eV/K]
    temp=np.divide(np.multiply(np.power(E,2),tailedAbsorptivity(E, Eg, A, gam, theta, Eshift=0.0, normalized=False)),(np.exp(np.divide((E-Dmu),(np.multiply(k,T))))-1))
    #temp=np.divide(1,np.exp(np.divide((E-Dmu),(np.multiply(k,T))))-1)
    if normalized == False:
        return temp
    elif normalized == True:
        return np.divide(temp,np.amax(temp))
    
    
def one_over(x):
    """Vectorized 1/x, treating x==0 manually"""
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1.23984193 / x[~near_zero]
    return x

inverse = one_over

"""
==================================================
===============SET THE VARIABLES==================
==================================================
"""
Temperature=temperatureExp[0]  #Temperature in Kelvin
BGEnergy=0.85
A=5
gam=0.006
DeltaMu=0.3
theta=1
EfShift=0.00
startCell=50
dataLength=140
absorptionLength=1

#energyData= energyData[startCell:dataLength]
#Normalized280K=Normalized280K[startCell:dataLength]
#Normalized4K=Normalized4K[startCell:dataLength]

"""
==================================================
===============GET THEORETICAL SPECTRA==================
==================================================
"""
#This is a scale I use to get a higher density of data points
energySimu = np.arange(0.6, 1.2, 0.01)
energySimu = energySimu.astype(float)

#Calculate arrays based on artificial high density data
calculated_FD_simu = fermiDirac(energySimu, BGEnergy, Temperature)
calculated_UT_simu = urbachTail(energySimu, BGEnergy, gam, theta, True)
calculated_DOS_simu= simpleDOS(energySimu, BGEnergy, A, normalized=True)
calculated_DOS_FD_simu = simpleDOS_FD(energySimu, BGEnergy, A, Temperature, normalized=True)
calculated_tDOS_simu = tailedDOS(energySimu, BGEnergy, A, gam, theta)
calculated_tDOS_FD_simu = tailedDOS_FD(energySimu, BGEnergy, A, Temperature, gam, theta, EfShift, normalized=True)
calculated_tAbsorptivity_simu = tailedAbsorptivity(energySimu, BGEnergy, A, gam, theta, EfShift,absorptionLength, normalized=True)
calculated_tailedAbsortivityOccCorr_simu = tailedAbsortivityOccCorr(energySimu, BGEnergy, A, Temperature, gam, DeltaMu, theta, EfShift, absorptionLength, normalized=True)
calculated_sAbsorptivity_simu = simpleAbsorptivity(energySimu, BGEnergy, A, EfShift,absorptionLength,  normalized=True)
calculated_LSW_tailedAbs_PL_simu = LSW_tailedAbs_PL(energySimu, BGEnergy, A, Temperature, gam, DeltaMu, absorptionLength,  theta, Eshift=0, normalized=True)
calculated_LSW_tailedAbsNoCorr_PL_simu = LSW_tailedAbsNoCorr_PL (energySimu, BGEnergy, A, Temperature, gam, DeltaMu, absorptionLength,  theta, Eshift=0, normalized=True)

"""
plt.figure(2)
#Plot fermi dirac
plt.plot(energySimu, calculated_FD_simu, 'b', label='Fermi Dirac')
#Plot simple DOS
plt.plot(energySimu, calculated_DOS_simu, 'g', label='DOS')
#Plot simple DOS multiplied with fermi dirac
plt.plot(energySimu, calculated_DOS_FD_simu, 'r', label='simple DOS FD')
#Plot urbach tail function
plt.plot(energySimu, calculated_UT_simu, 'y', label='UT')
#Plot urbach tail function convoluted with simple DOS
plt.plot(energySimu, calculated_tDOS_simu, 'b--', label='UT conv DOS')
#Plot tailed DOS multiplied with fermi dirac
plt.plot(energySimu, calculated_tDOS_FD_simu, 'r--', label='UT conv DOS FD')
#Plot simple absorptivity
#plt.plot(energySimu, calculated_sAbsorptivity_simu, 'b--', label='simple Abs')
#Plot tailed absorptivity
#plt.plot(energySimu, calculated_tAbsorptivity_simu, 'b', label='tailed Abs')
plt.legend(loc='right')
plt.axis([0.7, 1, 0, 1])
plt.show()
"""

"""
plt.figure(2)
plt.plot(energySimu, calculated_LSW_tailedAbs_PL_simu, 'r', label='LSW_tailedAbsOC')
plt.plot(energySimu, calculated_LSW_tailedAbsNoCorr_PL_simu, label='LSW_tailedAbs')
plt.plot(energySimu, calculated_tAbsorptivity_simu, 'y', label='tailedAbsty')
plt.plot(energySimu, calculated_tDOS_FD_simu, label='tDOS_FD')
#plt.plot(energySimu, calculated_tDOS_simu, 'r', label='UT conv DOS')
plt.plot(energySimu  ,calculated_tailedAbsortivityOccCorr_simu,'g', label='tailed Absty OC')
plt.axvline(x=BGEnergy,linewidth=2.0, dashes=(5,5))  #mark the bandgap
plt.legend(loc='best')
plt.axis([0.6, 1.2, 0 , 1])
plt.show()
"""


"""
==================================================
===============PREPARE FOR THE LOOP==================
==================================================
"""

#make a new directory of all the output of PL fit data
original_path = os.getcwd()
new_path = os.path.join(original_path, "PL_FIT")
# 'exist_ok=True' replaces the need for the try/except block
os.makedirs(new_path, exist_ok=True)
print(f"Saving results to: {new_path}")


#Check how many things I imported
importSizeTemp=np.size(fileNameTemp)
#Make result arrays existing out of zeros
results_sDOS_PL = np.zeros((importSizeTemp,4),dtype=float)
results_tDOS_PL = np.zeros((importSizeTemp,5),dtype=float)

# --- INITIALIZE STORAGE LISTS ---
# We will append dictionaries to these lists, then convert to DataFrame at the end
simpleDOS_results_list = []
tailedDOS_results_list = []

# Turn off interactive plotting to prevent windows popping up
plt.ioff()

"""
==================================================
===============FIT EXPERIMENTAL DATA==================
==================================================
"""

#Try fitting with original simple DOS model, and the more advanced model with an Urbach tail
BGEnergy_sDOS = BGEnergy
A_sDOS = A
BGEnergy_tDOS = BGEnergy
A_tDOS = A


#Loop to fit with the simple model
for i in range(0, importSizeTemp):
    
    current_temp = temperatureExp[i]
    current_file = fileNameTemp[i]
    print(f"Processing T = {current_temp}K ({i+1}/{len(fileNameTemp)})...")

    #1. LOAD DATA
    #select the data desired
    importedFileTemp=np.array(pd.read_csv(current_file, sep='\t', header=None))
    #Energy scale
    wavelength_raw = importedFileTemp[:, 0].astype(float)
    intensity_raw = importedFileTemp[:, 1].astype(float)
    #Convert wavelength in nm to energy in eV (Non-Linear Grid)
    with np.errstate(divide='ignore'): # Avoid division by zero if wavelength has 0s 
            energy_raw = 1239.84193 / wavelength_raw
    
    # Sort data so Energy is strictly increasing (Required for interpolation)
    sort_indices = np.argsort(energy_raw)
    energy_sorted = energy_raw[sort_indices]
    intensity_sorted = intensity_raw[sort_indices]

    # --- THE FUNDAMENTAL FIX: RESAMPLE TO UNIFORM GRID ---
    # Create a perfectly linear energy axis with the same number of points
    num_points = len(energy_sorted)
    energyData = np.linspace(energy_sorted.min(), energy_sorted.max(), num_points)
    
    # Interpolate intensity onto this new linear grid
    f_interp = interp1d(energy_sorted, intensity_sorted, kind='linear', fill_value="extrapolate") #create the interpolation function
    IntensityRaw = f_interp(energyData) #get the new intensity data based on the linear energy grid

    #2. SMOOTHEN AND NORMALIZE DATA
    NormalizedRaw =np.divide(IntensityRaw,np.amax(IntensityRaw))
    IntensityRaw_smooth = savgol_filter(IntensityRaw, 51, 2)  #Apply a Savitzky-Golay filter to smoothen the data
    IntensityRaw_smooth_norm =np.divide(IntensityRaw_smooth,np.amax(IntensityRaw_smooth))

    #Calculate arrays based on experimental energy array
    #calculatedDOS= simpleDOS(energyData, BGEnergy, 3)
    #calculatedDOS_FD = simpleDOS_FD(energyData, BGEnergy, 10, Temperature)
    calculated_FD = fermiDirac(energyData, BGEnergy, Temperature)
    calculated_UT = urbachTail(energyData, BGEnergy, gam, theta)
    calculated_tDOS = tailedDOS(energyData, BGEnergy, A, gam, theta)
    calculated_tDOS_FD = tailedDOS_FD(energyData, BGEnergy, A, Temperature, gam, theta, EfShift, normalized=False)

    #3. PLOT 1: RAW VS SMOOTHED
    plt.figure()
    plt.plot(energy_raw, intensity_raw, 'k.', markersize = 1, label='raw data original')
    plt.plot(energyData, IntensityRaw, 'r.', markersize = 1, label='raw data resampled')
    plt.plot(energyData, IntensityRaw_smooth, 'b.', markersize = 1, label='smoothed data')
    plt.xlim(0.6, 1.2)
    plt.title(f'Raw vs Smoothed Data at {current_temp}K')
    plt.xlabel('Energy (eV)')
    plt.ylabel('PL Intensity (a.u.)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(new_path, f'PL_{current_temp}K_1_Raw_Smooth.png'))
    plt.close() #close memory
    print(f"Saved {os.path.join(new_path, f'PL_{current_temp}K_1_Raw_Smooth.png')}")

    #4. FIT SIMPLE DOS MODEL
    p0_sDOS = [BGEnergy_sDOS, A_sDOS, current_temp]
        
    try:
        BestVal_sDOS_FD, _ = curve_fit(simpleDOS_FD, energyData, IntensityRaw_smooth, 
                                       p0=p0_sDOS, maxfev=5000)
        #fitted_sDOS_FD = simpleDOS_FD(energyData, *BestVal_sDOS_FD)
        fitted_sDOS_FD = simpleDOS_FD(energyData, BestVal_sDOS_FD[0], BestVal_sDOS_FD[1], BestVal_sDOS_FD[2])


        # Calculate R^2 to determine the goodness of fit
        ss_tot = np.sum((IntensityRaw_smooth - np.mean(IntensityRaw_smooth))**2)
        residuals_sDOS_model = IntensityRaw_smooth - fitted_sDOS_FD
        ss_res_sDOS = np.sum(residuals_sDOS_model**2)
        r_squared_sDOS = 1 - (ss_res_sDOS / ss_tot)

        #Determine the intensity by integrating the original
        PeakIntensity_raw=np.trapezoid(IntensityRaw_smooth, energyData)
        #Determine the intensity by integrating the fitted graph and multiplying with the height
        #PeakIntensity_DOS=np.multiply(np.multiply(np.trapezoid(fitted_DOS_FD_77K, -energyData),np.amax(Intensity77K_smooth)),np.amax(fitted_DOS_FD_77K))
        PeakIntensity_sDOS=np.trapezoid(fitted_sDOS_FD, energyData)
        
        # Store results
        simpleDOS_results_list.append({
            "T_meas": current_temp,
            "Eg": BestVal_sDOS_FD[0],
            "A": BestVal_sDOS_FD[1],
            "T_fit": BestVal_sDOS_FD[2],
            "R_squared": r_squared_sDOS,
            "PL_intensity_raw": PeakIntensity_raw,
            "PL_intensity_fit": PeakIntensity_sDOS
        })
        
        # Update guess for next loop (optional but helpful)
        BGEnergy_sDOS = BestVal_sDOS_FD[0]
        A_sDOS = BestVal_sDOS_FD[1]
        
    except Exception as e:
        print(f"Simple DOS Fit failed for {current_temp}K: {e}")
        p0_sDOS = np.zeros(3)
        fitted_sDOS_FD = np.zeros_like(energyData)

    #5. FIT TAILED DOS MODEL
    #p0_tDOS = [BGEnergy_tDOS, A_tDOS, current_temp, gam]
    p0_tDOS = [BGEnergy_tDOS, A_tDOS, current_temp, gam]

    # Define constraints to prevent crashes
    # [Eg, A, T, gam]
    lower_bounds = [0.5, 0, 10, 1e-5]  # gam must be > 0
    upper_bounds = [1.5, np.inf, np.inf, np.inf]

    try:
        BestVal_tDOS_FD, _ = curve_fit(tailedDOS_FD, energyData, IntensityRaw_smooth, 
                                       p0=p0_tDOS, 
                                       #bounds=(lower_bounds, upper_bounds), 
                                       maxfev=5000)
        #fitted_tDOS_FD = tailedDOS_FD(energyData, *BestVal_tDOS_FD)
        fitted_tDOS_FD = tailedDOS_FD(energyData, BestVal_tDOS_FD[0], BestVal_tDOS_FD[1], 
                                      BestVal_tDOS_FD[2], BestVal_tDOS_FD[3])


        # Calculate R^2 to determine the goodness of fit
        residuals_tDOS_model = IntensityRaw_smooth - fitted_tDOS_FD
        ss_res_tDOS = np.sum(residuals_tDOS_model**2)
        r_squared_tDOS = 1 - (ss_res_tDOS / ss_tot)

        # Integral
        PeakIntensity_tDOS=np.trapezoid(fitted_tDOS_FD, energyData)

        # Store results
        tailedDOS_results_list.append({
            "T_meas": current_temp,
            "Eg": BestVal_tDOS_FD[0],
            "A": BestVal_tDOS_FD[1],
            "T_fit": BestVal_tDOS_FD[2],
            "Gamma": BestVal_tDOS_FD[3],
            "R_squared": r_squared_tDOS,
            "PL_intensity_raw": PeakIntensity_raw,
            "PL_intensity_fit": PeakIntensity_tDOS
        })

        # Update guess for next loop (optional but helpful)
        BGEnergy_tDOS = BestVal_tDOS_FD[0]
        A_tDOS = BestVal_tDOS_FD[1]

    except Exception as e:
        print(f"Tailed DOS Fit failed for {current_temp}K: {e}")
        p0_tDOS = np.zeros(4)
        fitted_tDOS_FD = np.zeros_like(energyData)

    #6. PLOT 2: FITS AND RESIDUALS (Dual Axis)
    fig, ax = plt.subplots(layout='constrained')
    #Plot the measuredPL
    ax.plot(energyData, IntensityRaw_smooth, 'k.', markersize=2, alpha=0.3, label=f'{current_temp}K Experiment')

    #Plot the fitted result based on simple DOS FD
    ax.plot(energyData, fitted_sDOS_FD, 'g:', label='simple DOS fit')
    ax.fill_between(energyData, fitted_sDOS_FD.min(), fitted_sDOS_FD, facecolor="green", alpha=0.2)
    ax.plot(energyData, IntensityRaw_smooth-fitted_sDOS_FD, 'g--', label='residual sDOS_FD')

    # Plot the fitted results based on tailed DOS FD
    ax.plot(energyData, fitted_tDOS_FD, 'r:', label='tailed DOS fit')
    ax.fill_between(energyData, fitted_tDOS_FD.min(), fitted_tDOS_FD, facecolor="red", alpha=0.2)
    ax.plot(energyData, IntensityRaw_smooth-fitted_tDOS_FD, 'r--', label='residual tDOS_FD')
    #ax.fill_between(energyData, fitted_tailed, intensitySmooth, where=(intensitySmooth>fitted_tailed), facecolor='red', alpha=0.1, label='Residual')
    
    # We use transform=ax.transAxes so (0.05, 0.9) means 5% from left, 90% from bottom
    ax.text(0.05, 0.90, f'$R^2$ = {r_squared_sDOS:.4g}', 
            transform=ax.transAxes, color='green', fontsize=12, fontweight='bold')
    ax.text(0.05, 0.84, f'$R^2$ = {r_squared_tDOS:.4g}', 
            transform=ax.transAxes, color='red', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Normalized PL Intensity')
    ax.set_title(f'PL Fit Results: {current_temp}K')
    ax.legend(loc='upper right')
    ax.set_xlim(0.6, 1.2)
    #ax.set_ylim(-0.05, 1.1)

    #Second x-axis
    secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
    secax.set_xticks(np.arange(1.0,2.2,0.1))
    secax.set_xlabel('Wavelength (' + r'$\mu$' + 'm)')
    
    plt.savefig(os.path.join(new_path, f'PL_{current_temp}K_2_Fit_Result.png'))
    plt.close()
    print(f"Saved {os.path.join(new_path, f'PL_{current_temp}K_2_Fit_Result.png')}")


# --- SAVE TEXT FILES ---
# Create DataFrames
df_simple = pd.DataFrame(simpleDOS_results_list)
df_tailed = pd.DataFrame(tailedDOS_results_list)

# Define the single output file path
save_path_combined = os.path.join(new_path, "combined_fit_results.txt")

# Open the file in 'write' mode ('w')
with open(save_path_combined, 'w', newline='') as f:
# === PART 1: Simple DOS Table ===
    f.write("simpleDOS_fit\n") # Add the title line

    if not df_simple.empty:
            # Transpose immediately (Swap Rows & Columns)
            # 'T_meas' becomes the first row, just like 'Eg' is the second row
            df_simple_T = df_simple.transpose()
            
            # Save to the open file object 'f'
            # header=False: Removes the 0, 1, 2... column numbers
            # index=True:   Keeps the row names (T_meas, Eg, A...)
            df_simple_T.to_csv(f, sep='\t', header=False, index=True, float_format='%.4g')

    # === PART 2: Separator ===
    f.write("\n") # Add a blank line between tables
    
    # === PART 3: Tailed DOS Table ===
    f.write("tailedDOS_fit\n") # Add the title line
    
    if not df_tailed.empty:
        # Transpose immediately
        df_tailed_T = df_tailed.transpose()
        
        # Save to the same file
        df_tailed_T.to_csv(f, sep='\t', header=False, index=True, float_format='%.4g')

print(f"Saved merged results to: {save_path_combined}")

"""
# Transpose the DataFrames so Parameters are Rows (Index) and Temperatures are Columns
# We set 'T_meas' as the column header logic first
if not df_simple.empty:
    df_simple.set_index('T_meas', inplace=True)
    df_simple_T = df_simple.transpose() 
    # Save
    save_path_s = os.path.join(new_path, "simpleDOS_fit.txt")
    df_simple_T.to_csv(save_path_s, sep='\t')
    print(f"Saved {save_path_s}")

if not df_tailed.empty:
    df_tailed.set_index('T_meas', inplace=True)
    df_tailed_T = df_tailed.transpose()
    # Save
    save_path_t = os.path.join(new_path, "tailedDOS_fit.txt")
    df_tailed_T.to_csv(save_path_t, sep='\t')
    print(f"Saved {save_path_t}")
"""
    
# Restore interactive plotting if needed elsewhere
plt.ion()
print("All processing complete.")

fig = plt.figure(figsize=(10, 12), layout='constrained')

plt.subplot(2, 2, 1)
plt.plot(df_simple['T_meas'], df_simple['Eg'], 'go-', label='Simple DOS Eg')
plt.plot(df_tailed['T_meas'], df_tailed['Eg'], 'ro-', label='Tailed DOS Eg')
plt.xlabel('Temperature (K)')
plt.ylabel('Band Gap $E_g$ (eV)')
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.title('Fitted band gap')

plt.subplot(2, 2, 2)
plt.plot(df_simple['T_meas'], df_simple['T_fit'], 'g.', label='Simple DOS fitted temperature')
plt.plot(df_tailed['T_meas'], df_tailed['T_fit'], 'r.', label='Tailed DOS fitted temperature')
plt.plot(df_tailed['T_meas'], df_tailed['T_meas'], label='Lattice temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Temperature (K)')
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.title('Fitted temperature')

plt.subplot(2, 2, 3)
plt.plot(df_tailed['T_meas'], df_tailed['Gamma'], 'b.', label='Gamma')
plt.xlabel('Temperature (K)')
plt.ylabel('Gamma (eV)')
plt.legend(loc="best")
plt.title('Fitted gamma of tailed DOS model')

plt.subplot(2, 2, 4)
plt.loglog(np.divide(1, df_simple['T_meas']), df_simple['PL_intensity_raw'], 'ko-', markersize=5, label='Raw intensity')
plt.loglog(np.divide(1, df_simple['T_meas']), df_simple['PL_intensity_fit'], 'go-', markersize=5, label='sDOS fitted intensity')
plt.loglog(np.divide(1, df_tailed['T_meas']), df_tailed['PL_intensity_fit'], 'ro-', markersize=5, label='tDOS fitted intensity')
plt.xlabel('Inverse Temperature ($K^{-1}$)')
plt.ylabel('Integrated intensity (a.u.)')
plt.legend(loc="best")
plt.title('Arrhenius plot (log-log scale)')

plt.savefig(os.path.join(new_path, f'PL_fitting_parameters.png'))
plt.show()

os.chdir(path)