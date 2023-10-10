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



def generate_weights_from_SFHs_non_param(n,percen=True):
    """Create non-parametric SFHs with Dense Basis and obtain stellar mass percentiles

        Parameters
        ----------
        n: int
                Number of SFHs to generate
        percent: bool
                Return stellar mass percentiles or not
        Returns
        -------
        times: array, size 1.000
                Cosmic time values for each mass value in Gyr
        ms: array, size 1.000
                Normalized and non-cumulative mass curve as a function of time
        if percent is true; percentiles: array, size 9
            9 values for the time at which 10%, 20%, ... 90% of the total stellar mass are formed
        """


    priors = db.Priors() #load priors from Dense Basis

    curves=[]
    times=[] #needed because rand_time length and step may vary (depends on the redshift)

    z=np.zeros((n,)) #present time 

    
    for i in range(n):
        rand_sfh_tuple=priors.sample_sfh_tuple()
        #modify priors for the fractional sSFR in each time bin(dispersive - txs follow Dirichlet distribution of alpha=1)
        rand_sfh_tuple[3:]=np.cumsum(np.random.dirichlet(np.ones((3,)), size=1))
        rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = z[i]) 
        #save SFH
        curves.append(rand_sfh*1e9) #conversion from Msun/yr to Msun/Gyr
        times.append(rand_time)

    
    ms=[]
    #non-cumulative mass curves
    for index,curve in enumerate(curves):        
        sfr_0=curve
        m=[]
        t=times[index]
        #print(t)
        step=t[1]-t[0]
        #print(step)
        for i,tx in enumerate(t):  
             m_t=sfr_0[i]*step #this gives directly the mass curve (non-cumulative)
             m.append(m_t)
        ms.append(m/np.sum(m)) #normalized (weigths)

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
                percent.append(times[i][ind])
             percentiles.append(percent)  
        return np.array(times),np.array(ms),np.array(percentiles)
    else:
        return np.array(times),np.array(ms)

def get_tbins(dir_name,strs_1,strs_2):
    """Get age bins of MILES SSP spectra

        Parameters
        ----------
        dir_name: str
                Directory of MILES SSP spectra
        strs_1: str
                First part of the files' names
        strs_2: str
                Last part of the files' names
                
        Returns
        -------
        lib_n: array, size 53
                MILES time bins (lookback time) in Gyr
        """
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
    """Get metallicity bins of MILES SSP spectra

        Parameters
        ----------
        dir_name: str
                Directory of MILES SSP spectra
        strs_1: str
                First part of the files' names
        strs_2: str
                Last part of the files' names
                
        Returns
        -------
        lib_n: array, size 12
                MILES [M/H] bins
        """
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
    """Load MILES SSP spectra with different ages into arrays

        Parameters
        ----------
        dir_name: str
                Directory of MILES SSP spectra
        strs_1: str
                First part of the files' names
        strs_2: str
                Last part of the files' names
                
        Returns
        -------
        wave: array, size 4300 (length of MILES spectra)
            Wavelength in Angstrom
        data: array, size (number time bins: 53, length of MILES spectra: 4300)
            Spectra 
        """
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

    """Load MILES SSP spectra with different ages and metallicities into arrays

    Parameters
    ----------
    dir_name: str
            Directory of MILES SSP spectra
    interpolate: bool
            If interpolate the spectra to get more/different metallicity bins than in MILES library
    z: array, size equal to the number of metallicity bins required
            Equally-spaced [M/H] bins in which we interpolate the MILES spectra
            
    Returns
    -------
    wave: array, size 4300 (length of MILES spectra)
        Wavelength in Angstrom
    data_extended_met: array, size (number time bins: 53, length of MILES spectra: 4300, number of metallicity bins: len(z))
        Spectra with different ages and metallicities

    """

    

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
    """Interpolate MILES SSP spectra in metallicity

    Parameters
    ----------
    metallicity_bins: list of float, size 12
            [M/H] bins of MILES SSP spectra
    
    z: array, size equal to the number of metallicity bins required
            Equally-spaced [M/H] bins in which we interpolate the MILES spectra
            
    data: array, size (number time bins: 53, length of MILES spectra: 4300, number of metallicity bins in MILES: 12)
            MILES SSP spectra with different ages and metallicity
    Returns
    -------
    data_extended: array, size (number time bins: 53, length of MILES spectra: 4300, number of metallicity bins: len(z))
            MILES SSP spectra interpolated in metallicity

    """

    
    data_extended=np.zeros((len(data[:,0,0]),len(data[0,:,0]),len(z) ))
    for i in range(len(data[:,0,0])):
        for j in range(len(data[0,:,0])):
            x=np.interp(z,metallicity_bins,data[i,j,:])
            data_extended[i,j,:]=x
    return data_extended



def interpolate_t(tbins,t,data):
    """Interpolate MILES SSP spectra in ages

    Parameters
    ----------
    tbins: list of float, size 53
            Age bins of MILES SSP spectra
    
    t: array, size equal to the number of age bins required
            Equally-spaced age bins in which we interpolate the MILES spectra
            
    data: array, size (number time bins: 53, length of MILES spectra: 4300)
            MILES SSP spectra with different ages and fixed metallicity
    Returns
    -------
    data_extended: array, size (number time bins required: 1.000, length of MILES spectra: 4300)
            MILES SSP spectra interpolated in Ages with fixed metallicity

    """

    data_extended=np.zeros((len(t),len(data[0,:])))
    for i in range(4300):
        x=np.interp(t,tbins,data[:,i])
        data_extended[:,i]=x
    return data_extended


def create_spectrum(t,m,wave,data): #only for a galaxy at a time
    """Linear combination of MILES SSP spectra previously interpolated in age and metallicity with weights from the SFHs (obtaining spectra from composite stellar populations)

    Parameters
    ----------
    t: array, size equal to the number of age bins required, 1.000
        Equally-spaced age bins in which we interpolate the MILES spectra

    ms: array, size 1.000
                Normalized and non-cumulative mass curve as a function of time

    wave: array, size 4300 (length of MILES spectra)
            Wavelength in Angstrom

    data: array, size (1.000,4300)
        MILES SSP interpolated in time and metallicity

    Returns
    -------
    wave: array, size 4300 (length of MILES spectra)
            Wavelength in Angstrom

    sed: array, size 4300 (length of MILES spectra)
            Normalized CSP spectra
    """

    spectrum=[]
    for l in range(len(t)):  #we append older first
        spectrum.append(m[l]*data[-l]) #multiply by the weights
    #data is normalized, we do normalize the flux
    spectrum=np.array(spectrum)
    sed=np.sum(spectrum,axis=0) #we add the terms of the linear combination and normalize
    return wave,sed/np.median(sed)

def generate_all_spectrums(t,ms,wave,data_extended):
    """Perform several linear combinations of MILES SSP spectra previously interpolated in age and metallicity with weights from the SFHs (obtaining spectra from composite stellar populations)

    Parameters
    ----------
    t: array, size equal to the number of age bins required, 1.000
        Equally-spaced age bins in which we interpolate the MILES spectra

    ms: list of arrays, size (n: number of SFHs, len(t): 1.000)    
        List of normalized and non-cumulative mass curves as a function of time

    wave: array, size 4300 (length of MILES spectra)
        Wavelength in Angstrom

    data_extended: array, size (1.000,4300)
        MILES SSP interpolated in time and metallicity

    Returns
    -------
    wave: array, size 4300 (length of MILES spectra)
        Wavelength in Angstrom

    seds: list of arrays, size (n: number of SFHs, length of MILES spectra: 4300)
        List of normalized CSP spectra
    """
    seds=[]
    for m in ms[:]:
        wave,sed=create_spectrum(t,m,wave,data_extended)
        seds.append(sed)
    return wave,seds

def plot_sed_sfh(ms,t,wave,seds,n_int):
    """Check plot

    Parameters
    ----------
    ms: list of arrays, size (n: number of SFHs, len(t): 1.000)    
        List of normalized and non-cumulative mass curves as a function of time

    t: array, size equal to the number of age bins required, 1.000
        Equally-spaced age bins in which we interpolate the MILES spectra

    wave: array, size 4300 (length of MILES spectra)
        Wavelength in Angstrom

   n_int: int
        Step between spectra to plot

    """
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


    z= np.linspace(-2.3,0.4,15)

   
    #10.000 different SFH for each z
    print('Loading MILES spectra and interpolating in metallicity: ')
    tbins=get_tbins(dir_name='../MILES/MILES_BASTI_KU_baseFe',strs_1='Mku1.30Zp0.06T',strs_2='_iTp0.00_baseFe.fits')
    wave,data_met=get_data_met(dir_name='../MILES/MILES_BASTI_KU',interpolate=True,z=z)
    

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

    reshape=True
    save=True

    if reshape:
        print('Reshaping...') 
        seds=np.reshape(seds,(150000,4300)) #15 bins of metallicity x 10.000 SFHs = 150.000 different spectra
        percentiles=np.reshape(percentiles,(150000,9))
        zs=np.reshape(zs,(150000,))
        
        y=np.zeros((len(seds[:,0]),10))

        for i in range(len(seds[:,0])):
            y[i,:9]=percentiles[i,:]
            y[i,-1]=zs[i]
    
    if save:
        print('Saving...')
        np.save('../../seds_large/no_ssfr/seds.npy',seds) #file too large for github
        np.save('../../seds_large/no_ssfr/ms.npy',ms)  #file too large for github
        np.save('./saved_input/wave.npy',wave)
        np.save('./saved_input/t.npy',t[0])
        np.save('./saved_input/percent.npy',percentiles)
        np.save('./saved_input/zs.npy',zs)

        if reshape:
            np.save('./saved_input/y.npy',y) 
        
    
