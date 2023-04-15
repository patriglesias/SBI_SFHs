import numpy as np
import dense_basis as db
import matplotlib.pyplot as plt
import sys
from scipy import  integrate
from scipy import interpolate
import os
from astropy.io import fits
from tqdm import tqdm,trange

def generate_weights_from_SFHs_non_param(n,logMstar=10.0,z=0.0,percen=True):
    priors = db.Priors()
    curves=[]
    times=[] #not needed because if we fix z all rand_time are exactly the same
    for i in trange(n):
        rand_sfh_tuple=priors.sample_sfh_tuple()
        rand_sfh_tuple[0]=logMstar #logMstar at selected z (in this case z=0)
        rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = z) 
        curves.append(rand_sfh*1e-9) #conversion from Msun/yr to Msun/Gyr
        times.append(rand_time)
    
    ms=[]
    #non accumulative mass curves, we save it cause we will use it later
    for index,curve in tqdm(enumerate(curves)):        
        sfr_0=curve
        m=[]
        t=times[index]
        step=t[1]-t[0]
        for i,tx in enumerate(t):  
             m_t=sfr_0[i]*step #this gives directly the mass curve (non accumulative)
             m.append(m_t)
        ms.append(m/np.sum(m)) #normalized (weigths!!)

    if percen:
        #compute percentiles
        percentiles=[]
        for i,curve in tqdm(enumerate(curves)):
             mcurve=ms[i]
             m=[]
             percent=[]
             for j in range(len(mcurve)):
                m.append(np.sum(mcurve[:j+1]))
             for k in range(1,10):
                ind=np.argmin(abs(np.array(m)-k/10))
                percent.append(t[ind])
             percentiles.append(percent)  
        return np.array(times),np.array(ms),np.array(percentiles)
    else:
        return np.array(times),np.array(ms)

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
    
    #old
    """
        for j in range(len(lib_n)):
        globals() ['hdul'+str(j)]=fits.open(dir_name+'/'+strs_1+lib[j]+strs_2)
        data.append(np.array(globals()['hdul'+str(j)][0].data))
    """
    
    #new
    for j in range(len(lib_n)):
        hdul=fits.open(dir_name+'/'+strs_1+lib[j]+strs_2)
        data.append(np.array(hdul[0].data))
        if j != range(len(lib_n))[-1]:
            hdul.close()

    hdr=hdul[0].header
    wave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1']
    
    ind_sorted=np.argsort(lib_n)
    data=np.array(data,ndmin=2)
    data=data[ind_sorted,:]
    lib_n=lib_n[ind_sorted]
    hdul.close()
    return wave,data

def interpolate(tbins,t,data,nwave=4300):
    data_extended=np.zeros((len(t),len(data[0,:])))
    for i in range(nwave):
        x=np.interp(t,tbins,data[:,i])
        data_extended[:,i]=x
    return data_extended

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
    for m in tqdm(ms[:]):
        wave,sed=create_spectrum(t,m,wave,data_extended)
        seds.append(sed)
    return np.array(wave),np.array(seds)

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
    plt.ylim(0.0,0.005)
    plt.xlabel('Lookback time [Gyr]')
    plt.title('Mstar norm non acummulative')
    plt.show()


if __name__ == '__main__':

    
    # TEST 

    save=True
    plot=True
    
    print('Step 1/4...')
    t,ms,percentiles=generate_weights_from_SFHs_non_param(1000)
    print('Step 2/4...')
    wave,data=get_data(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    tbins=get_tbins(dir_name='../MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    print('Step 3/4...')
    data_extended=interpolate(tbins,t[0],data)
    print('Step 4/4...')
    wave,seds=generate_all_spectrums(t[0],ms,wave,data_extended)
    
    if plot:  
        plot_sed_sfh(ms,t[0],wave,seds,1) 
    if save:
        print('Saving...')
        np.save('seds_150_non_par.npy',seds)
        np.save('wave_150_non_par.npy',wave)
        np.save('t_150_non_par.npy',t[0])
        np.save('ms_150_non_par.npy',ms)
        np.save('percent_150_non_par.npy',percentiles)

    

    

    

    
