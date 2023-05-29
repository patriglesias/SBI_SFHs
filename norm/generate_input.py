# GENERATE_INPUT

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import  integrate
from scipy import interpolate
import os
from astropy.io import fits
from tqdm import tqdm,trange
import dense_basis as db


def generate_weights_from_SFHs_non_param(n,mfix=False,logMstar=10.0,z=0.0,percen=True):
    priors = db.Priors()
    curves=[]
    times=[] #not needed because if we fix z all rand_time are exactly the same
    for i in range(n):
        rand_sfh_tuple=priors.sample_sfh_tuple()
        if mfix:
            rand_sfh_tuple[0]=logMstar #logMstar at selected z (in this case z=0)
        rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = z) 
        curves.append(rand_sfh*1e-9) #conversion from Msun/yr to Msun/Gyr
        times.append(rand_time)
    
    ms=[]
    #non accumulative mass curves, we save it cause we will use it later
    for index,curve in enumerate(curves):        
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
        if j==0:
            hdul0=fits.open(dir_name+'/'+strs_1+lib[j]+strs_2)
        data.append(fits.open(dir_name+'/'+strs_1+lib[j]+strs_2)[0].data)

    hdr=hdul0[0].header
    wave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1']
    
    ind_sorted=np.argsort(lib_n)
    data=np.array(data,ndmin=2)
    data=data[ind_sorted,:]
    lib_n=lib_n[ind_sorted]
    return wave,data


def get_data_met(dir_name,interpolate=True,z=np.arange(-2.3,0.6,0.2)):

    data_met=np.zeros((53,4300,len(z)))

    a='0.00_baseFe'
    names,metallicity_bins=get_metallicities(dir_name+'_baseFe',strs_1='Mku1.30Z',strs_2='T00.0300_iTp'+a+'.fits')
    data_metallicities=[]

    for k,n in enumerate(names):
        wave,data=get_data(dir_name+'_baseFe',n[:14],'_iTp'+a+'.fits')
        data_metallicities.append(data)

    data_met=np.zeros((53,4300,len(metallicity_bins)))

    for i in range(len(metallicity_bins)):
        data_met[:,:,i]=data_metallicities[i]

    if interpolate:
        data_extended_met=interpolate_z(metallicity_bins,z,data_met)
    else:
        data_extended_met=data_met

    return wave,data_extended_met


def interpolate_z(metallicity_bins,z,data):
    #(53,4300,28)
    data_extended=np.zeros((len(data[:,0,0]),len(data[0,:,0]),len(z) ))
    for i in range(len(data[:,0,0])):
        for j in range(len(data[0,:,0])):
            x=np.interp(z,metallicity_bins,data[i,j,:])
            data_extended[i,j,:]=x
    return data_extended



def interpolate_t(tbins,t,data):
    data_extended=np.zeros((len(t),len(data[0,:])))
    for i in range(4300):
        x=np.interp(t,tbins,data[:,i])
        data_extended[:,i]=x
    return data_extended


def create_spectrum(t,m,wave,data): #only for a galaxy at a time
    spectrum=[]
    for l in range(len(t)):  #we append older first
        spectrum.append(m[l]*data[-l]) #multiply by the weights
    #data is normalized!! we do normalize the flux
    spectrum=np.array(spectrum)
    sed=np.sum(spectrum,axis=0) #we add the terms of the linear combination and normalize
    return wave,sed/np.median(sed)

def generate_all_spectrums(t,ms,wave,data_extended):
    seds=[]
    for m in ms[:]:
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


    #z= np.arange(-2.3,0.6,0.2)
    
    different=True

    if different:
   
        #10.000 different SFH for each z
        print('Loading MILES spectra and interpolating in metallicity: ')
        tbins=get_tbins(dir_name='../MILES/MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
        wave,data_met=get_data_met(dir_name='../MILES/MILES_BASTI_KU',interpolate=False)#z=z)

        seds=[]
        percentiles=[]
        ms=[]
        zs=[]
        
        n=10000 #number of SFHs for each z
    
        print('Generating 10.000 SFHs and their corresponding spectra for each Z:')
        for k,i in tqdm(enumerate(z)):
                print('z= ',k)
                t,m,per=generate_weights_from_SFHs_non_param(n)
                data_extended=interpolate_t(tbins,t[0],data_met[:,:,k])
                wave,sed=generate_all_spectrums(t[0],m,wave,data_extended)

                seds.append(sed)
                percentiles.append(per)
                ms.append(m)
                zs.append(np.ones((n,))*i)

    
    else:
        #10.000 equal SFHs for each Z
        print('Loading MILES spectra and interpolating in metallicity: ')
        tbins=get_tbins(dir_name='../MILES/MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
        wave,data_met=get_data_met(dir_name='../MILES/MILES_BASTI_KU_baseFe',z=z)
        
        n=10000 #number of SFHs for each Z
        t,m,per=generate_weights_from_SFHs_non_param(n)

        seds=[]
        percentiles=[]
        ms=[]
        zs=[]

        print('Generating 10.000 SFHs and their corresponding spectra for each Z:')
        for k,i in tqdm(enumerate(z)):
                print('z= ',k)
                data_extended=interpolate_t(tbins,t[0],data_met[:,:,k])
                wave,sed=generate_all_spectrums(t[0],m,wave,data_extended)

                seds.append(sed)
                percentiles.append(per)
                ms.append(m)
                zs.append(np.ones((n,))*i)

    
    reshape=True

    if reshape:
        print('Reshaping...')
        seds=np.reshape(seds,(120000,4300))
        percentiles=np.reshape(percentiles,(120000,9))
        zs=np.reshape(zs,(120000,))
        
        y=np.zeros((len(seds[:,0]),10))

        for i in range(len(seds[:,0])):
            y[i,:9]=percentiles[i,:]
            y[i,-1]=zs[i]
    
        np.save('./saved_input/y_12.npy',y) 
    
    save=True

    if save:
        print('Saving...')
        np.save('../../seds_large/norm/seds_12.npy',seds)
        np.save('./saved_input/wave_12.npy',wave)
        np.save('./saved_input/t_12.npy',t[0])
        np.save('../../seds_large/norm/ms_12.npy',ms)
        np.save('./saved_input/percent_12.npy',percentiles)
        np.save('./saved_input/zs_12.npy',zs)
        
    
