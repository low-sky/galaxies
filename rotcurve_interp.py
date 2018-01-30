import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage, misc, interpolate
from scipy.interpolate import BSpline, make_lsq_spline

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.table import Table
from spectral_cube import SpectralCube
from galaxies import Galaxy

from pandas import DataFrame, read_csv
import pandas as pd
import statsmodels.formula.api as smf


def rotcurve(name,smooth='False',knots=8):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    WARNING: Only for NGC1672 and M33 at the moment.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy that we care about.
    smooth : bool
        Determines whether the returned rotation
        curve returned is smoothed or not.
    knots : int
        Number of internal knots in BSpline of
        vrot, which is used to calculate epicyclic
        frequency.
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve.
    k : scipy.interpolate.interp1d
        Function for the interpolated epicyclic
        frequency.
    '''
    
    # Basic info
    gal = Galaxy(name)
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc
    
    # Rotation Curves
    if name=='NGC1672':
        fname = "phangsdata/NGC1672_co21_12m+7m+tp_RC.txt"
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
        R, vrot = np.loadtxt(fname,skiprows=True,unpack=True)
        # R = Radius from center of galaxy, in arcsec.
        # vrot = Rotational velocity, in km/s.
    elif name=='M33':
        m33 = pd.read_csv('phangsdata/m33_rad.out_fixed.csv')
        R = m33['r']
        vrot = m33['Vt']
    # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but both are treated as unitless!
    
    # Adding a (0,0) data point to rotation curve
    if R[0]!=0:
        R = np.roll(np.concatenate((R,[0]),0),1)
        vrot = np.roll(np.concatenate((vrot,[0]),0),1)
    
    # Units & conversions
    R = R*u.arcsec
    vrot = vrot*u.km/u.s    
    R = R.to(u.rad)            # Radius, in radians.
    R = (R*d).value            # Radius, in pc, but treated as unitless.
    
    
    
    # BSpline of vrot(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of vrot(R).
                                                            # 'vrot' is now a function, not an array.
    # Creating "higher-resolution" rotation curve
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    
    # SMOOTH BSpline of vrot(R)
    vrot_s = bspline(R,vrot(R),knots=knots,lowclamp=True)

    
    # Epicyclic Frequency
    dVdR = np.gradient(vrot_s(R),R)
    k2 =  2.*(vrot_s(R)**2 / R**2 + vrot_s(R)/R*dVdR)
    k = interpolate.interp1d(R,np.sqrt(k2))
    
    
    if smooth==True:
        return R, vrot_s, k
    else:
        return R, vrot, k


def rotmap(name):
    '''
    Returns "observed velocity" map, and "rotation
    map". (The latter is just to make sure that the
    code is working properly.)
    WARNING: Only works for NGC1672 at the moment.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy that we care about.
        
    Returns:
    --------
    vobs : np.ndarray
        Map of observed velocity, in km/s.
    R : np.ndarray
        Map of radii of galaxy, in pc.
    Dec, RA : np.ndarray
        2D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    gal = Galaxy(name)
    vsys = gal.vsys
    if name.upper()=='M33':
        vsys = gal.velocity
        # For some reason, M33's "Galaxy" object has velocity listed as "velocity" instead of "vsys".
    I = gal.inclination
    RA_cen = gal.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
    Dec_cen = gal.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
    PA = (gal.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
                                                     # NOTE: The x-direction is defined as the LoN.
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc


    # Header
    if name=='NGC1672':
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
    elif name=='M33':
        hdr = fits.getheader('phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')

    # vrot Interpolation
    R_1d, vrot, k_discard = rotcurve(name,smooth=0)  # Creates "vrot" interpolation function, and 1D array of R.


    # Generating displayable grids
    X,Y = gal.radius(header=hdr, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
    #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
    #               to the line of nodes.

    R = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    R = (R.value<R_1d.max()).astype(int) * R  
    R[ R==0 ] = np.nan                           # Grid of radius, with values outside interpolation range removed.

    skycoord = gal.skycoord_grid(header=hdr)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(R)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, R, Dec, RA


def bspline(X,Y,knots=8,k=3,lowclamp=False, highclamp=False):
    '''
    Returns a BSpline interpolation function
    of a provided 1D curve.
    With fewer knots, this will provide a
    smooth curve that ignores local wiggles.
    
    Parameters:
    -----------
    X,Y : np.ndarray
        1D arrays for the curve being interpolated.
    knots : int
        Number of INTERNAL knots, i.e. the number
        of breakpoints that are being considered
        when generating the BSpline.
    k : int
        Degree of the BSpline. Recommended to leave
        at 3.
    lowclamp : bool
        Enables or disables clamping at the lowest
        X-value.
    highclamp : bool
        Enables or disables clamping at the highest
        X-value.
        
    Returns:
    --------
    spl : scipy.interpolate._bsplines.BSpline
        Interpolation function that works over X's
        domain.
    '''
    
    # Creating the knots
    t_int = np.linspace(X.min(),X.max(),knots)  # Internal knots, incl. beginning and end points of domain.

    t_begin = np.linspace(X.min(),X.min(),k)
    t_end   = np.linspace(X.max(),X.max(),k)
    t = np.r_[t_begin,t_int,t_end]              # The entire knot vector.
    
    # Generating the spline
    w = np.zeros(X.shape)+1                     # Weights.
    if lowclamp==True:
        w[0]=X.max()*1000000                    # Setting a high weight for the X.min() term.
    if highclamp==True:
        w[-1]=X.max()*1000000                   # Setting a high weight for the X.max() term.
    spl = make_lsq_spline(X, Y, t, k,w)
    
    return spl

def localshear(name,knots=8):
    '''
    Returns the local shear parameter (i.e. the
    Oort A constant) for a galaxy with a provided
    rotation curve, based on Equation 4 in Martin
    & Kennicutt (2001).
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    knots : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve.
        
    Returns:
    --------
    A : scipy.interpolate._bsplines.BSpline
        Oort A "constant", as a function of 
        radius R.
    '''
    gal = Galaxy(name)
    
    # Use "interp" to generate R, vrot (smoothed).
    R, vrot, k_discard = gal.rotcurve(smooth=True, knots=knots)
    
    # Oort A constant.
    Omega = vrot(R) / R     # Angular velocity.
    dOmegadR = np.gradient(Omega,R)
    A = -1./2. * R*dOmegadR
    A = bspline(R[np.isfinite(A)],A[np.isfinite(A)],knots=999)
    
    return A

def linewidth_iso(name,knots=8):
    '''
    Returns the effective LoS velocity dispersion
    due to the galaxy's rotation, sigma_gal, for
    the isotropic case.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    knots : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve, which
        is used in calculation of epicyclic
        frequency (and, therefore, sigma_gal).
        
    Returns:
    --------
    R : np.ndarray
        Radius array.
    sigma_gal : scipy.interpolate._bsplines.BSpline
        Interpolation function for sigma_gal that
        works over R.
    '''
    gal = Galaxy(name)
    
    # Beam
    if name.upper()=='NGC1672':
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
    elif name.upper()=='M33':
        hdr = fits.getheader\
            ('phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')
            # WARNING: The IRAM .fits files give headers that galaxies.py misinterprets somehow,
            #    causing the galaxy to appear weirdly warped and lopsided.
            # WARNING: The peakvels.fits gives accurate data, but does not contain beam information.
            #    The IRAM .fits files contain beam information which should be interpreted correctly.
        hdr_beam = fits.getheader\
            ('notphangsdata/m33.co21_iram.14B-088_HI.mom0.fits')  # We're just using this one for the beam.
    
    # Beam width
    if name.upper()=='M33':
        # This is just because M33 needs one .fits files for headers to be properly interpreted, and another
        #    to provide beam information.
        beam = hdr_beam['BMAJ']*u.deg
    else:
        beam = hdr['BMAJ']*u.deg                     # Beam size, in degrees   
    beam = beam.to(u.rad)                       # Beam size, in radians
    d = (gal.distance).to(u.pc)
    Rc = beam*d / u.rad                         # Beam size, in parsecs
    
    # Use "interp" to generate R, vrot (smoothed), k.
    R, vrot, k = gal.rotcurve(smooth=1,knots=knots)
    
    # Calculate sigma_gal = kappa*Rc
    sigma_gal = k(R)*Rc

    # Removing nans and infs
    # (Shouldn't be anything significant-- just a "nan" at R=0.)
    index = np.arange(sigma_gal.size)
    R_clean = np.delete(R, index[np.isnan(sigma_gal)==True])
    sigma_gal_clean = np.delete(sigma_gal, index[np.isnan(sigma_gal)==True])
    sigma_gal = bspline(R_clean,sigma_gal_clean,knots=20)


    # Cubic Interpolation of sigma_gal
    #K=3     # Order of the BSpline
    #t,c,k = interpolate.splrep(R,sigma_gal,s=0,k=K)
    #sigma_gal_spline = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of sigma_gal(R).
    
    return R, sigma_gal

def moments(name,mode=''):
    '''
    Returns things like 'sigma' or 'Sigma' for
    a galaxy where the moment maps are provided.
    Basically, just the Warmup.py stuff.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    mode : str
        mode=='beam' : returns beam size, in pc.
        mode=='sigma': returns rad,sigma.
        mode=='Sigma': returns rad,Sigma.

    Returns:
    --------
    R : np.ndarray
        Radius array.
    sigma_gal : scipy.interpolate._bsplines.BSpline
        Interpolation function for sigma_gal that
        works over R.
    '''
    # Warmup.py stuff, originally for NGC1672.

    # Initializing
    if name.upper()=='NGC1672':
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
        I_mom0 = fits.getdata('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')  # In K km/s.
        I_mom1 = fits.getdata('phangsdata/ngc1672_co21_12m+7m+tp_mom1.fits') 
        I_max = fits.getdata('phangsdata/ngc1672_co21_12m+7m+tp_tpeak.fits')  # Peak temperature, in K.
    elif name.upper()=='M33':
        hdr = fits.getheader\
            ('phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')
            # WARNING: The IRAM .fits files give headers that galaxies.py misinterprets somehow,
            #    causing the galaxy to appear weirdly warped and lopsided.
            # WARNING: The peakvels.fits gives accurate data, but does not contain beam information.
            #    The IRAM .fits files contain beam information which should be interpreted correctly.
        hdr_beam = fits.getheader\
            ('notphangsdata/m33.co21_iram.14B-088_HI.mom0.fits')  # We're just using this one for the beam.
        I_mom0 = fits.getdata('notphangsdata/m33.co21_iram.14B-088_HI.mom0.fits')/1000. # Now in K km/s.
        I_mom1 = fits.getdata(\
                'phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')
        I_max = fits.getdata('notphangsdata/m33.co21_iram.14B-088_HI.peaktemps.fits')   # Peak T, in K.
    gal = Galaxy(name)
    rad = gal.radius(header=hdr)
    rad = rad.to(u.pc)                                      # Converts rad from Mpc to pc.
    d = gal.distance
    d = d.to(u.pc)                                          # Converts d from Mpc to pc.

        
    

    # Pixel sizes
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))*u.deg # The size of each pixel, in degrees. 
                                                                         # Ignore that third dimension; that's 
                                                                         # pixel size for the speed.
    print "Pixel size, in degrees = "+str(pixsizes_deg[0])
    pixsizes = pixsizes_deg[0].to(u.rad)                    # Pixel size, in radians.
    pcperpixel =  pixsizes.value*d                          # Number of parsecs per pixel.
    print "Pixel size, in parsecs = "+str(pcperpixel)
    pcperdeg = pcperpixel / pixsizes_deg[0]
    print "There are +"+str(pcperdeg)+"."

    # Beam width
    if name.upper()=='M33':
        # This is just because M33 needs one .fits files for headers to be properly interpreted, and another
        #    to provide beam information.
        beam = hdr_beam['BMAJ']
    else:
        beam = hdr['BMAJ']                                  # Beam size, in degrees
    beam = beam * pcperdeg                                  # Beam size, in pc

    # Gradient of mom1
    gradv = ndimage.sobel(I_mom1/pcperpixel)              # Sobel gradient of I_mom1; = grad(mean velocity), in km/s/pc
    gradvl = ndimage.sobel(I_mom1 * beam / pcperpixel)    # Sobel gradient of I_mom1 * beam width, in km/s

    # Line width, Surface density
    alpha = 6.7
    sigma = I_mom0 / (np.sqrt(2*np.pi * I_max))
    Sigma = alpha*I_mom0
    
    if mode=='beam':
        return beam
    elif mode=='sigma':
        return rad, sigma
    elif mode=='Sigma':
        return rad, Sigma
    else:
        print "SELECT A MODE."
