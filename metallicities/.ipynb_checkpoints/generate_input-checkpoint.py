# GENERATE_INPUT

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import  integrate
from scipy import interpolate
import os
from astropy.io import fits
from tqdm import tqdm


def escalon(t,ti):
    return t>ti

def sfr_linear_exp(t,tau,ti,mgal):
    i=integrate.quad(lambda t:(t-ti)*np.e**(-(t-ti)/tau)*escalon(t,ti),0,14)
    A=mgal/i[0]
    return A*(t-ti)*np.e**(-(t-ti)/tau)*escalon(t,ti) #units are Msun/Gyr

def generate_weights_from_SFHs(SFR,mgal=10**10,tau=np.linspace(0.3,5,100),ti=np.arange(0,5,0.5),tmin=0,tmax=14,step=0.01,percen=True):
    curves=[]
    t=np.arange(tmin,tmax+step,step) 
    for i in tau:
      for j in ti:
        curve=SFR(t,i,j,mgal)
        curves.append(curve)

    ms=[]
    #non accumulative mass curves, we save it cause we will use it later
    for index,curve in enumerate(curves):        
        sfr_0=curve
        m=[]
        for i,tx in enumerate(t):   
             m_t=sfr_0[i]*step #this gives directly the mass curve (non accumulative)
             m.append(m_t)
        ms.append(m/np.sum(m))

    if percen:
        #compute percentiles
        percentiles=[]
        for i,curve in enumerate(curves):
             mcurve=ms[i]
             m=[]
             percent=[]
             for j in range(len(mcurve)):
                m.append(np.sum(mcurve[:j+1]))
             for k in range(1,10):
                ind=np.argmin(abs(np.array(m)-k/10))
                percent.append(t[ind])
             percentiles.append(percent)  
        return t,ms,percentiles
    else:
        return t,ms

# +
def get_tbins(dir_name,strs_1,strs_2):
    library=os.listdir(dir_name)
    x,y=len(strs_1),len(strs_2)
    lib=[]
    lib_n=[]
    for i,j in enumerate(library):
        if j[:x]==strs_1:
            lib.append(j[x:y+1])
            lib_n.append(float(j[x:y+1]))     
    lib_n=np.array(sorted(lib_n))
    return lib_n

def get_metallicities(dir_name,strs_1,strs_2):
    library=os.listdir(dir_name)
    x,y=len(strs_1),len(strs_2)
    lib_n=[]
    names=[]
    for i,j in enumerate(library):
        if j[:x]==strs_1 and j[-y:]==strs_2:
            if j[x:y+1][0]=='m':
              lib_n.append(-float(j[x:-y][1:])) 
            elif j[x:y+1][0]=='p':
              lib_n.append(+float(j[x:-y][1:]))
            names.append(j)
            
    ind_sort=np.argsort(lib_n)
    names=np.array(names)[ind_sort]
    lib_n=np.array(lib_n)[ind_sort]
    return names,lib_n



# +
def get_data(dir_name,strs_1,strs_2):
    library=os.listdir(dir_name)
    x,y=len(strs_1),len(strs_2)
    lib=[]
    lib_n=[]
    for i,j in enumerate(library):
        if j[:x]==strs_1:
            lib.append(j[x:y+1])
            lib_n.append(float(j[x:y+1]))
        
    lib_n=np.array(lib_n)
    data=[]
    
    for j in range(len(lib_n)):
        globals() ['hdul'+str(j)]=fits.open(dir_name+'/'+strs_1+lib[j]+strs_2)
        data.append(np.array(globals()['hdul'+str(j)][0].data))

    hdr=hdul0[0].header
    wave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1']
    
    ind_sorted=np.argsort(lib_n)
    data=np.array(data,ndmin=2)
    data=data[ind_sorted,:]
    lib_n=lib_n[ind_sorted]
    return wave,data


def get_data_met(dir_name,z=np.arange(-2.3,0.4,0.1)):
    names,metallicity_bins=get_metallicities(dir_name,strs_1='Mku1.30Z',strs_2='T00.0300_iTp0.00_baseFe.fits')
    data_metallicities=[]

    for k,n in enumerate(names):
        wave,data=get_data(dir_name,n[:14],'_iTp0.00_baseFe.fits')
        data_metallicities.append(data)

    data_met=np.zeros((53,4300,12))
    
    for i in range(12):
        data_met[:,:,i]=data_metallicities[i]
        
    data_extended_met=interpolate_z(metallicity_bins,z,data_met)
    return data_extended_met

    

# +
def interpolate_z(metallicity_bins,z,data):
    #(53,4300,27)
    data_extended=np.zeros((len(data[:,0,0]),len(data[0,:,0]),len(z)))
    for i in range(len(data[:,0,0])):
        for j in range(len(data[0,:,0])):
            x=np.interp(z,metallicity_bins,data[i,j,:])
            data_extended[i,j,:]=x
    return data_extended


def interpolate_t(tbins,t,data):
    data_extended=np.zeros((len(t),len(data[0,:,0]),len(data[0,0,:])))
    for i in range(4300):
        for j in range(len(data[0,0,:])):
            x=np.interp(t,tbins,data[:,i,j])
            data_extended[:,i,j]=x
    return data_extended


# -

def create_spectrum(t,m,wave,data): #only for a galaxy at a time
    spectrum=[]
    for i in range(len(t)):  #we append older first
         spectrum.append(m[i]*data[-i]) #multiply by the weights
    #data is not normalized, we do not normalize the flux
    spectrum=np.array(spectrum)
    sed=np.sum(spectrum,axis=0) #we add the terms of the linear combination
    return wave,sed

def generate_all_spectrums(t,ms,wave,data_extended):
    seds=[]
    for i,m in tqdm(enumerate(ms[:])):
        wave,sed=create_spectrum(t,m,wave,data_extended)
        seds.append(sed)
    return wave,seds

def plot_sed_sfh(ms,t,wave,seds,n_int):
    for i,sed in enumerate(seds[::n_int]):
        plt.plot(wave,sed,alpha=0.7)
    plt.xlabel('Wavelenght [$\\AA$]')
    plt.title('Artificial spectrum')    
    plt.show()
    t_back=t[::-1]
    for i,m in enumerate(ms[::n_int]):
       plt.plot(t_back,m,'-')

    plt.xlim(14,0)
    plt.xlabel('Lookback time [Gyr]')
    plt.title('Mstar norm non acummulative')
    plt.show()


if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (20,15)
    params = {'xtick.labelsize': 20, 'ytick.labelsize': 20}
    plt.rcParams.update(params)
    plt.rcParams.update({'font.size': 22})

    # TEST (linear exp parametrization - 1000 curves - BASTI,Mku1.30,27 metallicities,BaseFe)
    
    t,ms,percentiles=generate_weights_from_SFHs(SFR=sfr_linear_exp,mgal=10**10,tau=np.logspace(-0.5,0.7,10),ti=np.linspace(0,5,10),tmin=0,tmax=14,step=0.01,percen=True)
    tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    data_met=get_data_met(dir_name='../MILES_BASTI_KU_baseFe',z=np.arange(-2.3,0.4,0.1))
    data_extended=interpolate_t(tbins,t,data_met)
    seds=np.zeros((1000,4300,27))

    for k,i in enumerate(z):
        wave,seds[:,:,k]=generate_all_spectrums(np.arange(0,14+0.01,0.01),ms,wave,data_extended[:,:,k])
        if k<3:
            plot_sed_sfh(ms,t,wave,seds[:,:,k],1) 
