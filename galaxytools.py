import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy.table import Table
from astropy.convolution import convolve_fft, Gaussian2DKernel
from spectral_cube import SpectralCube, Projection
from galaxies import Galaxy

import copy
import os


def info(name,conbeam=None):
    '''
    Returns basic info from galaxies.
    Astropy units are NOT attached to outputs.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy.
    conbeam : u.quantity.Quantity
        Width of the beam in pc or ",
        if you want the output to be
        convolved.
        
    Returns:
    --------
    hdr : fits.header.Header
        Header for the galaxy.
    beam : float
        Beam width, in deg.
    I_mom0 : np.ndarray
        0th moment, in K km/s.
    I_mom1 : np.ndarray
        Velocity, in km/s.
    I_tpeak : np.ndarray
        Peak temperature, in K.
    cube : SpectralCube
        Spectral cube for the galaxy.
    sfr : np.ndarray
        2D map of the SFR, in Msun/kpc^2/yr.
    '''
    name = name.lower()
    
    if name=='m33':
        hdr = fits.getheader(\
                'notphangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')
        hdr_beam = fits.getheader('notphangsdata/m33.co21_iram.14B-088_HI.mom0.fits')
            # WARNING: The IRAM .fits files give headers that galaxies.py misinterprets somehow,
            #    causing the galaxy to appear weirdly warped and lopsided.
            #    The peakvels.fits gives accurate data... but does not contain beam information,
            #    so the IRAM one is used for that.
        beam = hdr_beam['BMAJ']
        I_mom0 = fits.getdata('notphangsdata/m33.co21_iram.14B-088_HI.mom0.fits')/1000. # Now in K km/s.
        I_mom1 = fits.getdata(\
                'notphangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')/1000.
                # Now in km/s.
                # Technically not a moment1 map, but it works in this context. (???)
        I_tpeak = fits.getdata('notphangsdata/m33.co21_iram.14B-088_HI.peaktemps.fits')   # Peak T, in K.
    else:
        hdr = fits.getheader('phangsdata/'+name+'_co21_12m+7m+tp_mom0.fits')
        beam = hdr['BMAJ']                                                    # In degrees.
        I_mom0 = fits.getdata('phangsdata/'+name+'_co21_12m+7m+tp_mom0.fits')    # In K km/s.
        I_mom1 = fits.getdata('phangsdata/'+name+'_co21_12m+7m+tp_mom1.fits')    # In km/s.
        I_tpeak = fits.getdata('phangsdata/'+name+'_co21_12m+7m+tp_tpeak.fits')  # In K.
    
    # Fix the headers so WCS doesn't think that they're 3D!
    if name!='m33':
        # Should I find a more elegant way of telling which headers have 3D keywords?
        hdrcopy = copy.deepcopy(hdr)
        for kw in ['CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
            del hdrcopy[kw]
        for i in ['1','2','3']:
            for j in ['1', '2', '3']:
                del hdrcopy['PC0'+i+'_0'+j]
        hdr = hdrcopy

    # SFR map
    if name=='m33':
        sfr_map = Projection.from_hdu(fits.open('notphangsdata/cube.fits')[13])
    else:
        sfr_map = Projection.from_hdu(fits.open('phangsdata/'+name+'_sfr_fuvw4.fits'))
    sfr = sfr_map.reproject(hdr) # Msun/yr/kpc^2. See header.
                                 # https://www.aanda.org/articles/aa/pdf/2015/06/aa23518-14.pdf
           
    # Spectral Cube
    if name=='m33':
        cube = None
        print "M33 doesn't have a cube available."
    else:
        cube = SpectralCube.read('phangsdata/'+name+'_co21_12m+7m+tp_flat_round_k.fits')
        
        
    # CONVOLUTION, if enabled:
    if conbeam!=None:
        hdr,I_mom0, I_tpeak, cube = cube_moments(name,conbeam)    # Convolved moments, with their cube.
        sfr = sfr.reproject(hdr)                                  # The new header might have different res.!
        sfr = convolve_sfr(name,hdr,sfr,conbeam)                  # Convolved SFR map.
    else:
        sfr = sfr.value
    
    return hdr,beam,I_mom0,I_mom1,I_tpeak,cube,sfr

def cube_moments(name,conbeam):
    '''
    Extracts the mom0 and tpeak maps from
    a convolved data cube.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy.
    conbeam : float
        Convolution beam width, in pc 
        OR arcsec. Must specify units!
        
    Returns:
    --------
    hdrc : fits.header.Header
        Header for the galaxy's convolved
        moment maps.
    I_mom0c : np.ndarray
        0th moment, in K km/s.
    I_tpeakc : np.ndarray
        Peak temperature, in K.
    cubec : SpectralCube
        Spectral cube for the galaxy,
        convolved to the resolution indicated
        by "conbeam".
    '''
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
    else:
        raise ValueError("'conbeam' must have units of pc or arcsec.")
        
    # Read cube
    filename = 'phangsdata/cube_convolved/'+name.lower()+'_co21_12m+7m+tp_flat_round_k_'+conbeam_filename+'.fits'
    if os.path.isfile(filename):
        cubec = SpectralCube.read(filename)
    else:
        raise ValueError(filename+' does not exist.')
    cubec.allow_huge_operations=True
    I_mom0c = cubec.moment0().to(u.K*u.km/u.s)
    I_tpeakc = cubec.max(axis=0).to(u.K)
    hdrc = I_mom0c.header
    
    return hdrc,I_mom0c.value, I_tpeakc.value, cubec

def convolve_sfr(name,hdr,map2d,conbeam):
    '''
    Returns SFR map, convolved to a beam
    width "beam".
    
    Parameters:
    -----------
    name : str
        Name of the galaxy.
    hdr : fits.header.Header
        Header for the galaxy.
    map2d : np.ndarray
        The map (e.g. SFR) that needs to 
        be convolved.
    conbeam : float
        Convolution beam width, in pc 
        OR arcsec. Must specify units!
        The actual width of the Gaussian
        is conbeam/np.sqrt(8.*np.log(2)).

    Returns:
    --------
    map2d_convolved : np.ndarray
        The same map, convolved.
    '''
    gal = Galaxy(name.upper())
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        conbeam_width = conbeam.to(u.pc) / np.sqrt(8.*np.log(2))   # Beam width in pc.
        conbeam_angle = conbeam / gal.distance.to(u.pc) * u.rad
        conbeam_angle = conbeam_angle.to(u.deg)
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_angle = conbeam.to(u.deg) / np.sqrt(8.*np.log(2))
    else:
        raise ValueError("'conbeam' must have units of pc or arcsec.")
    
    
    # Convert beam width into pixels, then feed this into a Gaussian-generating function.
    
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))[0]*u.deg # The size of each pixel, in deg.
    conbeam_pixwidth = conbeam_angle / pixsizes_deg  # Beam width, in pixels.
    print "Pixel width of beam: "+str(conbeam_pixwidth)+" pixels."
    
    #gauss = gaussian(beam_pixwidth)
    gauss = Gaussian2DKernel(conbeam_pixwidth)
    map2d_convolved = convolve_fft(map2d,gauss,normalize_kernel=True)
    return map2d_convolved

def convolve_cube(gal,cube,conbeam):
    '''
    Convolves a cube over a given beam, and
    then generates and returns the moment
    maps. The convolved cube is saved as well.
    
    Parameters:
    -----------
    gal : galaxies.galaxies.Galaxy
        "Galaxy" object for the galaxy.
    cube : SpectralCube
        Spectral cube for the galaxy.
    conbeam : float
        Beam width, in pc OR arcsec.
        Must specify units!
            
    Returns:
    --------
    cubec : SpectralCube
        Spectral cube for the galaxy,
        convolved to the resolution indicated
        by "conbeam".
    '''
    if conbeam.unit in {u.pc, u.kpc, u.Mpc}:
        conbeam_width = conbeam.to(u.pc)                     # Beam width in pc.
        conbeam_angle = conbeam / gal.distance.to(u.pc) * u.rad
        conbeam_angle = conbeam_angle.to(u.arcsec)
        conbeam_filename = str(conbeam.to(u.pc).value)+'pc'
    elif conbeam.unit in {u.arcsec, u.arcmin, u.deg, u.rad}:
        conbeam_angle = conbeam.to(u.arcsec)                 # Beam width in arcsec.
        conbeam_filename = str(conbeam.to(u.arcsec).value)+'arcsec'
    else:
        raise ValueError("'beam' must have units of pc or arcsec.")
    
    bm = Beam(major=conbeam_angle,minor=conbeam_angle)    # Actual "beam" object, used for convolving cubes
    print bm
    
    # Convolve the cube!
    cube = cube.convolve_to(bm)
    
    # Never convolve the cube again!
    filename = 'phangsdata/cube_convolved/'+name.lower()+'_co21_12m+7m+tp_flat_round_k_'+conbeam_filename+'.fits'
    print filename
    if os.path.isfile(filename):
        os.remove(filename)
        print filename+" has been overwritten."
    cube.write(filename)
    
    return cube

def gaussian(beam_pixwidth):
#    ____  ____   _____  ____  _      ______ _______ ______ 
#   / __ \|  _ \ / ____|/ __ \| |    |  ____|__   __|  ____|
#  | |  | | |_) | (___ | |  | | |    | |__     | |  | |__   
#  | |  | |  _ < \___ \| |  | | |    |  __|    | |  |  __|  
#  | |__| | |_) |____) | |__| | |____| |____   | |  | |____ 
#   \____/|____/|_____/ \____/|______|______|  |_|  |______|
#
# astropy's Gaussian2DKernel does the same job, except better and with more options.

    '''
    Returns a square 2D Gaussian centered on
    x=y=0, for a galaxy "d" pc away.
    
    Parameters:
    -----------
    beam : float
        Desired width of gaussian, in pc.
    d : float
        Distance to galaxy, in pc.
        
    Returns:
    --------
    gauss : np.ndarray
        2D Gaussian with width "beam".    
    '''
    axis = np.linspace(-4*beam_pixwidth,4*beam_pixwidth,int(beam_pixwidth*8))
    x, y = np.meshgrid(axis,axis)
    d = np.sqrt(x*x+y*y)
    
    sigma, mu = beam_pixwidth, 0.0
    g = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) / (sigma*np.sqrt(2.*np.pi)))
    return g
