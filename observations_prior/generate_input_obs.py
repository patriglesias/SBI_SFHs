
# GENERATE_INPUT for observations (prepare MILES spectra to compare with observations)

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

def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating  template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.
    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig
    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2
    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]
    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel
    conv_spectrum = np.sum(a*gau, 0)
    return conv_spectrum

def fwhm(l): #units: amstrong, from LaBarbera13
    #valid only for l in [4000,6000] A
    if  l<5500:
        return 0.0001*l+1.75
    elif l>=5500:
        return 0.0014*l-5.4

def prepare_data(w,f):
    #convolve for a sigma like 300 km/s, clip between 4023,6000 A and interpolate to get delta_lambda=1 A

    #CONVOLVE
    cvel=300000 #km/s
    psize=0.9 #miles delta lambda
    fluxes_conv_miles=[]
    sigma_miles=cvel * psize / w #km/s

    fwhm_sdss=np.ones((len(w)))
    for i in range(len(w)):
        fwhm_sdss[i]=fwhm(w[i])

    sigma_sdss=fwhm_sdss/(2.*np.sqrt(2.*np.log(2.)))

    max_sigma=np.sqrt(300**2+sigma_sdss**2-sigma_miles**2)
    FWHM_gal = 2.*np.sqrt(2.*np.log(2.)) * max_sigma / cvel * w
    FWHM_dif = np.sqrt(FWHM_gal**2)
    sigma = FWHM_dif/(psize*2.*np.sqrt(2.*np.log(2.))) # Sigma difference in pixels

    spec_conv=[]
    for i in range(len(f)):
        spec_conv.append(gaussian_filter1d(f[i], sigma))

    """#CLIP (maybe unnecessary if we interpolate later??)
    ind=np.where((w>4022.9)&(w<6000.1))[0]
    wave_short=w[ind]

    spec_conv_short=[]
    for i in range(len(f)):
        spec_conv_short.append(spec_conv[i][ind])"""

    #INTERPOLATE (clip is intrisec)

    fluxes_MILES=np.zeros((53,len(np.arange(4023,6001,1))))

    for i in range(53):
        fluxes_MILES[i,:]=np.interp(np.arange(4023,6001,1),w,spec_conv[i])
    return np.arange(4023,6001,1), fluxes_MILES #notice here the flux is not normalized yet







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

    wave,data=prepare_data(wave,data) #flux not norm but convolved, clipped and interpolated
    return wave,data


def get_data_met(dir_name,z=np.arange(-2.3,0.6,0.2)):

    data_met=np.zeros((53,1978,len(z)))

    a='0.00_baseFe'
    names,metallicity_bins=get_metallicities(dir_name+'_baseFe',strs_1='Mku1.30Z',strs_2='T00.0300_iTp'+a+'.fits')
    data_metallicities=[]

    for k,n in enumerate(names):
        wave,data=get_data(dir_name+'_baseFe',n[:14],'_iTp'+a+'.fits')
        data_metallicities.append(data)

    data_met=np.zeros((53,1978,len(metallicity_bins)))

    for i in range(len(metallicity_bins)):
        data_met[:,:,i]=data_metallicities[i]

    data_extended_met=interpolate_z(metallicity_bins,z,data_met)

        
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
    for i in range(1978):
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


    z= np.arange(-1.5,0.5,0.1)
    


    different=True

    if different:
   
        #10.000 different SFH for each z
        print('Loading MILES convolved spectra and interpolating in metallicity: ')
        tbins=get_tbins(dir_name='../MILES/MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
        wave,data_met=get_data_met(dir_name='../MILES/MILES_BASTI_KU',z=z)

        seds=[]
        percentiles=[]
        ms=[]
        zs=[]
        
        n=10000 #number of SFHs for each z
        #n=5
        print('Generating 10.000 SFHs and their corresponding spectra for each Z:')
        for k,i in tqdm(enumerate(z)):
                print('z= ',k)
                t,m,per=generate_weights_from_SFHs_non_param(n,mfix=True,logMstar=14)
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
        t,m,per=generate_weights_from_SFHs_non_param(n,mfix=True,logMstar=14) #prior on the mass

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
        seds=np.reshape(seds,(200000,1978))
        percentiles=np.reshape(percentiles,(200000,9))
        zs=np.reshape(zs,(200000,))
        
        y=np.zeros((len(seds[:,0]),10))

        for i in range(len(seds[:,0])):
            y[i,:9]=percentiles[i,:]
            y[i,-1]=zs[i]
    
        np.save('./saved_input/y.npy',y) 
    
    save=True

    if save:
        print('Saving...')
        np.save('../../seds_large/obs/seds.npy',seds)
        np.save('./saved_input/wave.npy',wave)
        np.save('./saved_input/t.npy',t[0])
        np.save('../../seds_large/obs/ms.npy',ms)
        np.save('./saved_input/percent.npy',percentiles)
        np.save('./saved_input/zs.npy',zs)
        
    
